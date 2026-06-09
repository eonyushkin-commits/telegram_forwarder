import asyncio
import html as html_lib
import json
import logging
import os
import re

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from telethon import TelegramClient, events
from telethon.tl.types import Channel

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# User API (Telethon) — для чтения каналов
API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]
SESSION_PATH = os.environ.get("SESSION_PATH", "/session/forwarder")

# Bot API — для публикации
BOT_TOKEN = os.environ["BOT_TOKEN"]
DEST_CHAT = os.environ["DESTINATION_CHANNEL"]
SOURCES = [s.strip().lstrip("@") for s in os.environ["SOURCE_CHANNELS"].split(",") if s.strip()]

ADD_SOURCE_LINK = os.environ.get("ADD_SOURCE_LINK", "true").lower() == "true"

TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
SAFE_CHUNK = 3900

# ---------------------------------------------------------------------------
# AI-клиент (Google Gemini через OpenAI-совместимый API)
# ---------------------------------------------------------------------------
ai_client = AsyncOpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
ai_semaphore = asyncio.Semaphore(5)

SYSTEM_PROMPT = """Ты — AI-ассистент, обрабатывающий посты из Telegram-каналов о футболе. Анализируй текст и возвращай результат строго в формате JSON с тремя ключами: "is_target", "is_ad", "text".

Правила анализа:

1. "is_ad" (boolean): true, если пост является рекламным или спамом:
   - Призывы подписаться на чужой канал ("переходи по ссылке", "подписывайся", "читай в закрепе").
   - Реклама ставок, казино, прогнозов на спорт (даже если упоминается Локомотив).
   - Продажа товаров, курсов, билетов с сомнительных сайтов.
   - Теги #реклама, #промо.
   Если пост рекламный — "is_target" всегда false.

2. "is_target" (boolean): true, если в тексте упоминается футбольный клуб "Локомотив" (Москва) — его матчи, игроки, тренеры, трансферы, статистика, турнирное положение. Учитывай даже мимолётное упоминание или упоминание как часть большого материала на другую тему.

3. "text" (string):
   - Если "is_target" = false или "is_ad" = true: верни пустую строку "".
   - Если "is_target" = true и "is_ad" = false: перепиши пост по следующим правилам:
     а) Обязательно придумай и добавь в самое начало короткий информативный заголовок, отражающий суть новости.
     б) Сохраняй смысл, факты и все важные подробности оригинала максимально близко к тексту.
     в) Исправляй орфографические и пунктуационные ошибки.
     г) Оставляй важный бэкграунд, детали переговоров, информацию об интересе других клубов и ключевые причины событий (например, почему трансфер не состоялся раньше или позицию самого игрока). При этом удаляй исключительно жесткую вкусовщину автора, пустые эмоциональные эпитеты, иронию и риторические вопросы.
     д) Цитаты (прямая речь, слова игроков, тренеров) оставляй без изменений — допускается только исправление ошибок.
     е) ИГНОРИРУЙ информацию, вообще не связанную с контекстом ФК «Локомотив» (например, подробный разбор игры других команд). Однако, если упоминание других клубов или игроков объясняет контекст ситуации вокруг «Локомотива» (например, конкуренция за трансфер), эту информацию нужно оставить.
     ж) СТРОГО ЗАПРЕЩЕНО добавлять что-либо от себя (кроме заголовка), додумывать факты или менять суть инсайдов.
     з) ФОРМАТИРОВАНИЕ ЗАГОЛОВКА: Заголовок должен быть выделен жирным шрифтом с помощью HTML-тегов <b> и </b>. После заголовка обязательно сделай перенос строки с помощью символа \n\n, чтобы отделить его от основного текста.
     и) Текст должен оставаться связным, информативным и живым, сохраняя всю фактологическую глубину исходного поста, без искусственного сокращения важных деталей.

Верни ТОЛЬКО валидный JSON, без markdown-разметки (без ```json) и лишних слов.
Пример ответа: {"is_target": true, "is_ad": false, "text": "<b>Заголовок</b>\n\nТекст поста"}"""


def _sanitize_json(raw: str) -> str:
    """Убирает управляющие символы (кроме \n \r \t) которые ломают json.loads."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)


async def analyze_post(text: str) -> dict:
    if not text or len(text.strip()) < 10:
        return {"is_target": False, "is_ad": False, "text": ""}
    async with ai_semaphore:
        try:
            response = await ai_client.chat.completions.create(
                model="gemini-3.1-flash-lite",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text[:4000]},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            content = (response.choices[0].message.content or "").strip()
            if not content:
                choice = response.choices[0]
                logger.warning(
                    f"Gemini вернул пустой content. "
                    f"finish_reason={choice.finish_reason}, "
                    f"model={response.model}, "
                    f"usage={response.usage}, "
                    f"raw_message={choice.message}"
                )
                raise ValueError(f"Empty content (finish_reason={choice.finish_reason})")
            result = json.loads(_sanitize_json(content))
            return {
                "is_target": bool(result.get("is_target", False)),
                "is_ad": bool(result.get("is_ad", False)),
                "text": str(result.get("text", "")),
            }
        except Exception as e:
            logger.error(f"Ошибка AI-запроса (пост пропущен): {e}")
            return {"is_target": False, "is_ad": False, "text": ""}


# ---------------------------------------------------------------------------
# Подготовка текста к отправке
# ---------------------------------------------------------------------------
_ALLOWED_TAGS = ("b", "strong", "i", "em", "u", "s")


def html_safe(text: str) -> str:
    escaped = html_lib.escape(text, quote=False)
    for tag in _ALLOWED_TAGS:
        escaped = escaped.replace(f"&lt;{tag}&gt;", f"<{tag}>")
        escaped = escaped.replace(f"&lt;/{tag}&gt;", f"</{tag}>")
    return escaped


def strip_tags(text: str) -> str:
    return re.sub(r"</?[a-zA-Z][^>]*>", "", text)


def split_text(text: str, limit: int = SAFE_CHUNK) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks, cur = [], ""
    for line in text.split("\n"):
        if cur and len(cur) + len(line) + 1 > limit:
            chunks.append(cur)
            cur = ""
        if len(line) > limit:
            if cur:
                chunks.append(cur)
                cur = ""
            for i in range(0, len(line), limit):
                chunks.append(line[i: i + limit])
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------------------
# Публикация через Bot API (с обработкой rate limit 429)
# ---------------------------------------------------------------------------
async def _tg(http: httpx.AsyncClient, method: str, payload: dict, timeout: int = 60) -> dict:
    for _ in range(4):
        resp = await http.post(f"{TG_API}/{method}", json=payload, timeout=timeout)
        data = resp.json()
        if data.get("ok"):
            return data
        retry_after = data.get("parameters", {}).get("retry_after")
        if retry_after:
            logger.warning(f"Rate limit, ждём {retry_after}с…")
            await asyncio.sleep(retry_after + 1)
            continue
        return data
    return {"ok": False, "description": "retry limit"}


async def _send_chunk(http: httpx.AsyncClient, raw_chunk: str) -> dict:
    data = await _tg(
        http,
        "sendMessage",
        {
            "chat_id": DEST_CHAT,
            "text": html_safe(raw_chunk),
            "parse_mode": "HTML",
        },
    )
    if not data.get("ok") and "parse" in str(data.get("description", "")).lower():
        logger.warning("HTML не распарсился — отправляю чистым текстом")
        data = await _tg(
            http,
            "sendMessage",
            {
                "chat_id": DEST_CHAT,
                "text": strip_tags(raw_chunk),
            },
        )
    return data


async def publish(http: httpx.AsyncClient, text: str, source_link: str) -> bool:
    if ADD_SOURCE_LINK:
        text = (text + f"\n\nИсточник: {source_link}") if text else source_link
    if not text:
        text = source_link

    ok_all = True
    for chunk in split_text(text):
        data = await _send_chunk(http, chunk)
        if not data.get("ok"):
            logger.error(f"Bot API отказал для {source_link}: {data.get('description')}")
            ok_all = False
    return ok_all


# ---------------------------------------------------------------------------
# Главный запуск: Telethon слушает, httpx публикует
# ---------------------------------------------------------------------------
async def main():
    logger.info(f"🚀 Старт. Источники: {SOURCES}")

    tg_client = TelegramClient(SESSION_PATH, API_ID, API_HASH)
    await tg_client.start()
    logger.info("✅ Telethon: авторизация успешна")

    async with httpx.AsyncClient(follow_redirects=True) as http:
        # Проверка доступности канала-приёмника при старте
        check = await _tg(http, "getChat", {"chat_id": DEST_CHAT})
        if check.get("ok"):
            logger.info(f"✅ Канал-приёмник найден: {check['result'].get('title', DEST_CHAT)}")
        else:
            logger.error(
                f"❌ Канал-приёмник недоступен ({DEST_CHAT}): {check.get('description')}. "
                f"Убедитесь, что бот добавлен администратором в канал."
            )

        @tg_client.on(events.NewMessage(chats=SOURCES))
        async def handler(event):
            text = event.message.text or ""
            chat = await event.get_chat()
            username = getattr(chat, "username", None) or str(chat.id)
            msg_id = event.message.id
            source_link = f"https://t.me/{username}/{msg_id}"

            logger.info(f"📨 Новый пост от @{username} (id={msg_id}): {text[:80]!r}")

            verdict = await analyze_post(text)

            if verdict["is_ad"]:
                logger.info(f"🚫 По теме, но реклама — пропуск {source_link}")
                return

            if not verdict["is_target"]:
                logger.info(f"⏭  Не по теме — пропуск {source_link}")
                return

            logger.info(f"✅ Совпадение, постим {source_link}")
            publish_text = verdict["text"] if verdict["text"] else text
            await publish(http, publish_text, source_link)
            await asyncio.sleep(1)

        logger.info("👂 Слушаю новые сообщения...")
        await tg_client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
