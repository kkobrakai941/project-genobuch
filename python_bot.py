from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Tuple

import aiosqlite
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    filters,
)

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
except ImportError:
    pipeline = None

@dataclass(slots=True, frozen=True)
class Settings:
    token: str = os.getenv("BOT_TOKEN", "")
    openai_key: str | None = os.getenv("OPENAI_API_KEY")
    db_file: str = os.getenv("DB_FILE", "chatlog.sqlite")
    hf_model: str = os.getenv("HF_MODEL", "facebook/bart-large-cnn")
    use_local: bool = os.getenv("USE_LOCAL_SUMMARY", "0") == "1"

    @property
    def can_use_openai(self) -> bool:
        return bool(self.openai_key)

CFG = Settings()
if not CFG.token:
    raise RuntimeError("BOT_TOKEN env variable is required")
if not CFG.can_use_openai and not CFG.use_local:
    raise RuntimeError("No summarisation backend configured")
if CFG.can_use_openai:
    import openai
    openai.api_key = CFG.openai_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("summarybot")

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS events ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT,"
    "chat INTEGER,"
    "user INTEGER,"
    "name TEXT,"
    "payload TEXT,"
    "kind TEXT,"
    "ts INTEGER)"
)

class Store:
    def __init__(self, db_file: str):
        self._db_file = db_file

    async def init(self) -> None:
        async with aiosqlite.connect(self._db_file) as db:
            await db.execute(SCHEMA)
            await db.commit()

    @asynccontextmanager
    async def _connection(self):
        async with aiosqlite.connect(self._db_file) as db:
            yield db

    async def add(self, row: Tuple[int, int, str, str, str, int]) -> None:
        q = (
            "INSERT INTO events (chat,user,name,payload,kind,ts) "
            "VALUES (?,?,?,?,?,?)"
        )
        async with self._connection() as db:
            await db.execute(q, row)
            await db.commit()

    async def fetch(self, chat: int, hours: int) -> List[Tuple[str, str, str]]:
        now = int(time.time())
        start = now - hours * 3600
        q = (
            "SELECT name,payload,kind FROM events "
            "WHERE chat=? AND ts BETWEEN ? AND ? ORDER BY ts"
        )
        async with self._connection() as db:
            async with db.execute(q, (chat, start, now)) as cur:
                return await cur.fetchall()

store = Store(CFG.db_file)

_URL_RE = re.compile(r"http[s]?://\S+")
_JUNK_RE = re.compile(r"[^\w\s,.!?]")

def _clean(text: str) -> str:
    text = _URL_RE.sub("", text)
    return _JUNK_RE.sub("", text).strip()

class BaseSummariser:
    async def run(self, rows: Iterable[Tuple[str, str, str]]) -> str:
        raise NotImplementedError

class OpenAISummariser(BaseSummariser):
    async def run(self, rows):
        prepared = [f"{u}: {_clean(t)}" for u, t, k in rows if k == "text" and _clean(t)]
        if not prepared:
            return "No text messages to summarise."
        prompt = "Summarise this group chat:\n" + "\n".join(prepared)
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.6,
        )
        return resp.choices[0].message.content.strip()

class LocalSummariser(BaseSummariser):
    def __init__(self, model_name: str):
        if pipeline is None:
            raise RuntimeError("transformers is not available")
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._pipe = pipeline("summarization", model=mdl, tokenizer=tok)

    async def run(self, rows):
        corpus = "\n".join(
            f"{u}: {_clean(t)}" for u, t, k in rows if k == "text" and _clean(t)
        )
        if not corpus:
            return "No text messages to summarise."
        chunks: List[str] = []
        for i in range(0, len(corpus), 3500):
            chunk = corpus[i : i + 3500]
            s = self._pipe(chunk, max_new_tokens=130, do_sample=False)[0][
                "summary_text"
            ].strip()
            chunks.append(s)
        final = self._pipe(" ".join(chunks), max_new_tokens=150, do_sample=False)[0][
            "summary_text"
        ].strip()
        return final

summariser: BaseSummariser
if CFG.can_use_openai and not CFG.use_local:
    summariser = OpenAISummariser()
else:
    summariser = LocalSummariser(CFG.hf_model)

_FONT = "arial.ttf"
_PANEL_SIZE = (300, 200)

def _create_stub_panel(idx: int) -> Path:
    img = Image.new("RGB", _PANEL_SIZE, (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    ))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(_FONT, 24)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 80), f"Panel {idx}", fill=(0, 0, 0), font=font)
    path = Path(f"panel_{idx}.png")
    img.save(path)
    return path

def build_pdf(summary: str, chat: int, hrs: int) -> Path:
    pdf_path = Path(f"summary_{chat}_{int(time.time())}.pdf")
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    w, h = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(w / 2, h - 50, f"Group summary — last {hrs}h")
    c.setFont("Helvetica", 11)
    text_obj = c.beginText(40, h - 80)
    for line in summary.split("\n"):
        text_obj.textLine(line)
    c.drawText(text_obj)
    y = text_obj.getY() - 40
    for i in range(1, 4):
        stub = _create_stub_panel(i)
        c.drawImage(str(stub), 40 + (i - 1) * 180, y - 220, width=160, height=110)
        stub.unlink(missing_ok=True)
    c.showPage()
    c.save()
    return pdf_path

async def collect(update: Update, ctx: CallbackContext.DEFAULT_TYPE) -> None:
    if update.effective_chat and update.effective_chat.type in {"group", "supergroup"}:
        msg = update.effective_message
        name = (msg.from_user.username or msg.from_user.first_name or "anon")
        payload = msg.text or msg.caption or ""
        row = (
            update.effective_chat.id,
            msg.from_user.id,
            name,
            payload,
            msg.effective_attachment and msg.effective_attachment.type
            if hasattr(msg, "effective_attachment")
            else msg.effective_attachment,
            msg.date.timestamp(),
        )
        await store.add(row)

aSYNC_RANGE = range(1, 25)

async def summary_cmd(update: Update, ctx: CallbackContext.DEFAULT_TYPE) -> None:
    if update.effective_chat.type not in {"group", "supergroup"}:
        await update.effective_message.reply_text("This command works only in groups.")
        return
    try:
        hrs = int(ctx.args[0])
    except (IndexError, ValueError):
        await update.effective_message.reply_text("Usage: /summary <hours 1‑24>")
        return
    if hrs not in aSYNC_RANGE:
        await update.effective_message.reply_text("Hours must be between 1 and 24.")
        return
    await update.effective_message.reply_text("Generating summary… this might take a moment.")
    history = await store.fetch(update.effective_chat.id, hrs)
    if not history:
        await update.effective_message.reply_text("No messages found for that period.")
        return
    summary = await summariser.run(history)
    pdf_path = build_pdf(summary, update.effective_chat.id, hrs)
    await ctx.bot.send_message(update.effective_chat.id, summary)
    await ctx.bot.send_document(update.effective_chat.id, pdf_path.open("rb"))
    pdf_path.unlink(missing_ok=True)

aSYNC_START_TEXT = (
    "Add me to any group and type /summary <hours>. I’ll summarise the last N hours "
    "(1‑24) of conversation and send you a PDF!"
)

async def start_cmd(update: Update, _: CallbackContext.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(aSYNC_START_TEXT)

async def main() -> None:
    Path(CFG.db_file).parent.mkdir(parents=True, exist_ok=True)
    await store.init()
    app = (
        ApplicationBuilder()
        .token(CFG.token)
        .rate_limiter(None)
        .build()
    )
    app.add_handler(MessageHandler(filters.ALL, collect))
    app.add_handler(CommandHandler("summary", summary_cmd))
    app.add_handler(CommandHandler(["start", "help"], start_cmd))
    log.info("Bot is up — listening for updates…")
    await app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    asyncio.run(main())
