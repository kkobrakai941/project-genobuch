from __future__ import annotations

import os
import re
import time
import random
import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import telebot
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

USE_HF = os.getenv("USE_LOCAL_SUMMARY") == "1"
if USE_HF:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
else:
    import openai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("summarybot")

load_dotenv()


@dataclass(frozen=True, slots=True)
class Config:
    token: str = os.getenv("BOT_TOKEN", "")
    openai_key: str | None = os.getenv("OPENAI_API_KEY")
    db_path: str = os.getenv("DB_PATH", "chat_history.db")
    model_name: str = os.getenv("HF_MODEL_NAME", "facebook/bart-large-cnn")

    @property
    def use_openai(self) -> bool:
        return bool(self.openai_key)


CFG = Config()
if not CFG.token:
    raise RuntimeError("BOT_TOKEN not provided")
if not CFG.use_openai and not USE_HF:
    raise RuntimeError("No summarization backend")
if CFG.use_openai:
    openai.api_key = CFG.openai_key

class HistoryStore:
    _schema = (
        "CREATE TABLE IF NOT EXISTS messages ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "chat_id INTEGER, user_id INTEGER, username TEXT,"
        "body TEXT, mtype TEXT, ts INTEGER)"
    )

    def __init__(self, db_file: str):
        self._db = db_file
        self._init()

    def _connect(self):
        return sqlite3.connect(self._db, check_same_thread=False)

    def _init(self):
        with closing(self._connect()) as conn, conn, conn.cursor() as cur:
            cur.execute(self._schema)

    def add(self, chat: int, user: int, uname: str, body: str, mtype: str, ts: int):
        with closing(self._connect()) as conn, conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (chat_id,user_id,username,body,mtype,ts) VALUES (?,?,?,?,?,?)",
                (chat, user, uname, body, mtype, ts),
            )

    def fetch(self, chat: int, hours: int) -> Iterable[Tuple[str, str, str]]:
        end = int(time.time())
        start = end - hours * 3600
        with closing(self._connect()) as conn, conn, conn.cursor() as cur:
            cur.execute(
                "SELECT username, body, mtype FROM messages WHERE chat_id=? AND ts BETWEEN ? AND ? ORDER BY ts",
                (chat, start, end),
            )
            yield from cur.fetchall()


store = HistoryStore(CFG.db_path)

class Summarizer:
    url_re = re.compile(r"http[s]?://\\S+")
    junk_re = re.compile(r"[^\\w\\s,.!?]")

    @staticmethod
    def _clean(txt: str) -> str:
        txt = Summarizer.url_re.sub("", txt)
        return Summarizer.junk_re.sub("", txt).strip()

    def run(self, history: Iterable[Tuple[str, str, str]]) -> str:
        raise NotImplementedError


class OpenAISummarizer(Summarizer):
    def run(self, history):
        bodies = [
            f"{u}: {self._clean(t)}"
            for u, t, tp in history
            if tp == "text" and self._clean(t)
        ]
        if not bodies:
            return "No text messages."
        prompt = "Summarize this group chat:\\n\\n" + "\\n".join(bodies)
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.6,
        )
        return resp.choices[0].message.content.strip()


class LocalSummarizer(Summarizer):
    def __init__(self, model_name: str):
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._pipe = pipeline("summarization", model=mdl, tokenizer=tok)

    def run(self, history):
        text = "\\n".join(
            f"{u}: {self._clean(t)}"
            for u, t, tp in history
            if tp == "text" and self._clean(t)
        )
        if not text:
            return "No text messages."
        out = []
        for i in range(0, len(text), 3500):
            chunk = text[i : i + 3500]
            s = self._pipe(chunk, max_new_tokens=130, do_sample=False)[0][
                "summary_text"
            ].strip()
            out.append(s)
        return self._pipe(" ".join(out), max_new_tokens=150, do_sample=False)[0][
            "summary_text"
        ].strip()


summarizer: Summarizer = (
    LocalSummarizer(CFG.model_name) if USE_HF else OpenAISummarizer()
)

def _panel_stub(n: int) -> Path:
    img = Image.new(
        "RGB",
        (300, 200),
        (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        ),
    )
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    d.text((10, 80), f"Panel {n}", fill=(0, 0, 0), font=font)
    fn = Path(f"panel_{n}.png")
    img.save(fn)
    return fn

def build_pdf(summary: str, comic: str, chat_id: int, hrs: int) -> Path:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Summary {hrs}h", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, summary)
    pdf.ln(4)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Comic description", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, comic)
    pdf.ln(4)
    for i in range(1, 4):
        p = _panel_stub(i)
        pdf.image(str(p), w=60)
        p.unlink()
    out = Path(f"summary_{chat_id}_{int(time.time())}.pdf")
    pdf.output(str(out))
    return out

tg = telebot.TeleBot(CFG.token, parse_mode="HTML")

@tg.message_handler(content_types=["text", "photo", "sticker", "video", "document", "audio"])
def _collector(msg):
    if msg.chat.type in {"group", "supergroup"}:
        uname = msg.from_user.username or msg.from_user.first_name or "anon"
        body = msg.text or msg.caption or ""
        store.add(
            msg.chat.id,
            msg.from_user.id,
            uname,
            body,
            msg.content_type,
            msg.date,
        )

@tg.message_handler(commands=["summary"])
def _summary_cmd(msg):
    if msg.chat.type not in {"group", "supergroup"}:
        tg.reply_to(msg, "Group only.")
        return
    try:
        hrs = int(msg.text.split()[1])
        if not 1 <= hrs <= 24:
            raise ValueError
    except (IndexError, ValueError):
        tg.reply_to(msg, "Usage: /summary <1-24>")
        return
    tg.reply_to(msg, "Working…")
    hist = list(store.fetch(msg.chat.id, hrs))
    if not hist:
        tg.reply_to(msg, "No messages.")
        return
    summ = summarizer.run(hist)
    comic = (
        "Comic description disabled (offline)"
        if not CFG.use_openai
        else "Comic placeholder"
    )
    pdf_path = build_pdf(summ, comic, msg.chat.id, hrs)
    tg.send_message(msg.chat.id, summ)
    with pdf_path.open("rb") as f:
        tg.send_document(msg.chat.id, f, caption=f"{hrs}h summary")
    pdf_path.unlink()

@tg.message_handler(commands=["start", "help"])
def _start(msg):
    tg.reply_to(msg, "Add me to a group and use /summary <hours>.")

if __name__ == "__main__":
    Path(CFG.db_path).parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            log.info("Polling…")
            tg.infinity_polling(timeout=30, long_polling_timeout=30)
        except Exception as exc:
            log.error("polling error: %s", exc)
            time.sleep(5)
