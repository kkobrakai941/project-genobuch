import os, time, re, random, sqlite3, logging
from contextlib import closing
import telebot
import openai
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger(__name__)

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not BOT_TOKEN or not OPENAI_API_KEY:
    raise RuntimeError("BOT_TOKEN or OPENAI_API_KEY missing")

bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")
openai.api_key = OPENAI_API_KEY

DB_PATH = "chat_history.db"

def init_db():
    with closing(sqlite3.connect(DB_PATH)) as c, c, c.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, chat_id INTEGER, user_id INTEGER, username TEXT, text TEXT, ttype TEXT, ts INTEGER)"
        )

def save_msg(chat_id, user_id, username, text, ttype, ts):
    with closing(sqlite3.connect(DB_PATH)) as c, c, c.cursor() as cur:
        cur.execute(
            "INSERT INTO messages (chat_id, user_id, username, text, ttype, ts) VALUES (?,?,?,?,?,?)",
            (chat_id, user_id, username, text, ttype, ts),
        )

def load_msgs(chat_id, hours):
    now = int(time.time())
    start = now - hours * 3600
    with closing(sqlite3.connect(DB_PATH)) as c, c, c.cursor() as cur:
        cur.execute(
            "SELECT username, text, ttype FROM messages WHERE chat_id=? AND ts BETWEEN ? AND ? ORDER BY ts",
            (chat_id, start, now),
        )
        return cur.fetchall()

url_re = re.compile(r"http[s]?://\S+")

def clean(text):
    if not text:
        return ""
    text = url_re.sub("", text)
    return re.sub(r"[^\w\s,.!?]", "", text).strip()

def gpt(prompt, max_tokens, temp):
    r = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temp,
    )
    return r.choices[0].message.content.strip()

def make_summary(history):
    parts = [f"{u}: {clean(t)}" for u, t, tp in history if tp == "text" and clean(t)]
    if not parts:
        return "No text messages."
    prompt = "Summarize this group chat:\n\n" + "\n".join(parts)
    return gpt(prompt, 1200, 0.6)

def make_comic(summary):
    prompt = "Describe a 3‑panel comic for this summary:\n\n" + summary
    return gpt(prompt, 400, 0.8)

def panel_img(n):
    img = Image.new("RGB", (300, 200), (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        f = ImageFont.load_default()
    d.text((10, 80), f"Panel {n}", (0,0,0), f)
    path = f"panel_{n}.png"
    img.save(path)
    return path

def pdf_file(summary, comic, chat_id, hours):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Summary {hours}h", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, summary)
    pdf.ln(4)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Comic description", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, comic)
    pdf.ln(4)
    for i in range(1, 4):
        p = panel_img(i)
        pdf.image(p, w=60)
        os.remove(p)
    name = f"summary_{chat_id}_{int(time.time())}.pdf"
    pdf.output(name)
    return name

@bot.message_handler(content_types=["text", "photo", "sticker", "video", "document", "audio"])
def all_msgs(m):
    if m.chat.type in ("group", "supergroup"):
        u = m.from_user.username or m.from_user.first_name or "user"
        save_msg(m.chat.id, m.from_user.id, u, m.text or m.caption or "", m.content_type, m.date)

@bot.message_handler(commands=["summary"])
def cmd_sum(m):
    if m.chat.type not in ("group", "supergroup"):
        bot.reply_to(m, "Group only.")
        return
    try:
        h = int(m.text.split()[1])
    except (IndexError, ValueError):
        bot.reply_to(m, "Usage: /summary <1‑24>")
        return
    if not 1 <= h <= 24:
        bot.reply_to(m, "1‑24 only.")
        return
    bot.reply_to(m, "Working…")
    hist = load_msgs(m.chat.id, h)
    if not hist:
        bot.reply_to(m, "No messages.")
        return
    s = make_summary(hist)
    c = make_comic(s)
    pdf = pdf_file(s, c, m.chat.id, h)
    bot.send_message(m.chat.id, s)
    with open(pdf, "rb") as f:
        bot.send_document(m.chat.id, f, caption=f"{h}h summary")
    os.remove(pdf)

@bot.message_handler(commands=["start", "help"])
def cmd_start(m):
    bot.reply_to(m, "Add me to a group and use /summary <hours>.")

if __name__ == "__main__":
    init_db()
    while True:
        try:
            bot.infinity_polling(timeout=30, long_polling_timeout=30)
        except Exception as e:
            log.error("restart: %s", e)
            time.sleep(5)
