import os
import re
import json
import html as html_lib
import logging
import asyncio

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import AsyncOpenAI

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

BOT_TOKEN = os.environ["BOT_TOKEN"]
DEST_CHAT = os.environ["DESTINATION_CHANNEL"]
SOURCES = [s.strip().lstrip("@") for s in os.environ["SOURCE_CHANNELS"].split(",") if s.strip()]

POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "120"))
ADD_SOURCE_LINK = os.environ.get("ADD_SOURCE_LINK", "true").lower() == "true"

TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
SAFE_CHUNK = 3900             # режем сырой текст с запасом — экранирование удлиняет его

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
            result = json.loads(content)
            return {
                "is_target": bool(result.get("is_target", False)),
                "is_ad": bool(result.get("is_ad", False)),
                "text": str(result.get("text", "")),
            }
        except Exception as e:
            logger.error(f"Ошибка AI-запроса (пост пропущен): {e}")
            return {"is_target": False, "is_ad": False, "text": ""}


# ---------------------------------------------------------------------------
# Парсинг публичного канала через t.me/s/<channel>
# ---------------------------------------------------------------------------
def parse_channel(html: str, channel: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    posts: dict[int, dict] = {}

    for bubble in soup.select("div.tgme_widget_message[data-post]"):
        m = re.search(r"/(\d+)$", bubble.get("data-post", ""))
        if not m:
            continue
        post_id = int(m.group(1))

        text_el = bubble.select_one(".tgme_widget_message_text")
        text = text_el.get_text("\n", strip=True) if text_el else ""

        if post_id in posts:
            if text and not posts[post_id]["text"]:
                posts[post_id]["text"] = text
        else:
            posts[post_id] = {
                "id": post_id, "text": text,
                "link": f"https://t.me/{channel}/{post_id}",
            }

    return [posts[k] for k in sorted(posts)]


async def fetch_latest(http: httpx.AsyncClient, channel: str) -> list[dict]:
    url = f"https://t.me/s/{channel}"
    try:
        r = await http.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        return parse_channel(r.text, channel)
    except Exception as e:
        logger.error(f"Не удалось получить @{channel}: {e}")
        return []


# ---------------------------------------------------------------------------
# Подготовка текста к отправке
# ---------------------------------------------------------------------------
# Разрешённые Telegram inline-теги, которые мы хотим сохранить (AI ставит <b>).
_ALLOWED_TAGS = ("b", "strong", "i", "em", "u", "s")


def html_safe(text: str) -> str:
    """Экранирует <, >, & во всём тексте, затем возвращает разрешённые теги."""
    escaped = html_lib.escape(text, quote=False)  # & < >  ->  &amp; &lt; &gt;
    for tag in _ALLOWED_TAGS:
        escaped = escaped.replace(f"&lt;{tag}&gt;", f"<{tag}>")
        escaped = escaped.replace(f"&lt;/{tag}&gt;", f"</{tag}>")
    return escaped


def strip_tags(text: str) -> str:
    """Убирает все теги — для отправки чистым текстом, если HTML не распарсился."""
    return re.sub(r"</?[a-zA-Z][^>]*>", "", text)


def split_text(text: str, limit: int = SAFE_CHUNK) -> list[str]:
    """Бьёт длинный текст на части по границам строк, не разрывая слова без нужды."""
    if len(text) <= limit:
        return [text]
    chunks, cur = [], ""
    for line in text.split("\n"):
        if cur and len(cur) + len(line) + 1 > limit:
            chunks.append(cur)
            cur = ""
        if len(line) > limit:                       # одна строка длиннее лимита — режем жёстко
            if cur:
                chunks.append(cur)
                cur = ""
            for i in range(0, len(line), limit):
                chunks.append(line[i:i + limit])
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------------------
# Публикация в канал через Bot API (только текст, с обработкой rate limit 429)
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
    """Шлёт один кусок как HTML; если Telegram не смог распарсить разметку — шлёт чистым текстом."""
    data = await _tg(http, "sendMessage", {
        "chat_id": DEST_CHAT,
        "text": html_safe(raw_chunk),
        "parse_mode": "HTML",
    })
    if not data.get("ok") and "parse" in str(data.get("description", "")).lower():
        logger.warning("HTML не распарсился — отправляю чистым текстом")
        data = await _tg(http, "sendMessage", {
            "chat_id": DEST_CHAT,
            "text": strip_tags(raw_chunk),
        })
    return data


async def publish(http: httpx.AsyncClient, post: dict) -> bool:
    text = post.get("text") or ""
    if ADD_SOURCE_LINK:
        text = (text + f"\n\nИсточник: {post['link']}") if text else post["link"]
    if not text:
        text = post["link"]

    ok_all = True
    for chunk in split_text(text):
        data = await _send_chunk(http, chunk)
        if not data.get("ok"):
            logger.error(f"Bot API отказал для {post['link']}: {data.get('description')}")
            ok_all = False
    return ok_all


# ---------------------------------------------------------------------------
# Обработка новых постов: AI-фильтр → публикация → обновление позиции
# ---------------------------------------------------------------------------
async def process_posts(http, state: dict, channel: str, posts: list[dict]) -> None:
    for post in posts:
        verdict = await analyze_post(post["text"])
        if verdict["is_target"] and not verdict["is_ad"]:
            logger.info(f"✅ Совпадение, постим {post['link']}")
            publish_post = dict(post)
            if verdict["text"]:
                publish_post["text"] = verdict["text"]
            if not await publish(http, publish_post):
                break
            await asyncio.sleep(1)
        elif verdict["is_target"] and verdict["is_ad"]:
            logger.info(f"🚫 По теме, но реклама — пропуск {post['link']}")
        else:
            logger.info(f"⏭  Не по теме — пропуск {post['link']}")
        state[channel] = max(state.get(channel, 0), post["id"])


# ---------------------------------------------------------------------------
# Главный цикл
# ---------------------------------------------------------------------------
async def main():
    state: dict[str, int] = {}
    logger.info(f"🚀 Старт. Источники: {SOURCES}. Опрос каждые {POLL_INTERVAL}с.")

    async with httpx.AsyncClient(follow_redirects=True) as http:
        while True:
            for channel in SOURCES:
                posts = await fetch_latest(http, channel)
                if not posts:
                    continue

                if channel not in state:
                    state[channel] = posts[-1]["id"]
                    logger.info(f"@{channel}: первый запуск, стартовая позиция id={state[channel]}")
                    continue

                new_posts = [p for p in posts if p["id"] > state[channel]]
                if new_posts:
                    logger.info(f"@{channel}: новых постов — {len(new_posts)}")
                    await process_posts(http, state, channel, new_posts)

            await asyncio.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())