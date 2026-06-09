"""
server.py  —  A.I. WARZONE  —  True combined launcher
══════════════════════════════════════════════════════
Single Flask process on port 5000 serving both games
with their COMPLETE, unmodified game logic.

URL structure
─────────────
GET  /                          → index.html  (hub: intro + mode selection)
GET  /game1                     → index1.html (DeCorroDE standalone page)
GET  /game2                     → index2.html (City Sweep standalone page)
GET  /intro.mp4                 → intro video

── GAME 1 routes  (/game1/api/*) ──────────────────────
GET  /game1/api/stats
GET  /game1/api/nexeons
POST /game1/api/nexeon/select
POST /game1/api/new_round
POST /game1/api/guess
POST /game1/api/hint
POST /game1/api/skip
POST /game1/api/chatbot/activate
POST /game1/api/chatbot/chat
POST /game1/api/reset_progress
POST /game1/api/save_progress
POST /game1/api/load_progress

── GAME 2 routes  (/game2/api/*) ──────────────────────
GET  /game2/api/health
POST /game2/api/new_game
POST /game2/api/click
POST /game2/api/flag
POST /game2/api/anomaly_detect
GET  /game2/api/stats
POST /game2/api/reset

Directory layout expected
──────────────────────────
project/
├── server.py           ← this file
├── index.html          ← hub (intro + mode selection)
├── index1.html         ← DeCorroDE full page
├── index2.html         ← City Sweep full page
├── intro.mp4           ← optional intro video
│
├── data/               ← Game 1 data (same folder as before)
│   ├── rag_knowledge_base.json
│   ├── sentences.json
│   ├── player_progress.json
│   ├── nexeons.json
│   └── numpy_gan_weights.pkl
│
├── models/             ← Game 2 models
│   ├── gan_weights.pt
│   └── vae_weights.pt
│
├── hint_generator.py   ← optional Game 1 hint module
├── chatbot.py          ← optional Game 1 chatbot module
├── gan.py              ← Game 2 GAN inference
└── vae.py              ← Game 2 VAE inference
"""

# ═══════════════════════════════════════════════════════════════════
#  SHARED IMPORTS
# ═══════════════════════════════════════════════════════════════════
from __future__ import annotations
import json, os, random, re, pickle, sys, traceback
import datetime

import numpy as np
from flask import Flask, request, jsonify, send_from_directory, Response

HERE = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=HERE)

# ═══════════════════════════════════════════════════════════════════
#  SHARED ROUTES  —  hub page + game pages + intro video
# ═══════════════════════════════════════════════════════════════════

@app.route("/")
def serve_hub():
    return send_from_directory(HERE, "index.html")

@app.route("/game1")
@app.route("/game1/")
def serve_game1_page():
    return send_from_directory(HERE, "index1.html")

@app.route("/game2")
@app.route("/game2/")
def serve_game2_page():
    return send_from_directory(HERE, "index2.html")

@app.route("/intro.mp4")
def serve_intro_video():
    video_path = os.path.join(HERE, "intro.mp4")
    if not os.path.exists(video_path):
        return "", 404
    range_header = request.headers.get("Range", None)
    file_size    = os.path.getsize(video_path)
    if range_header:
        byte_range = range_header.replace("bytes=", "").split("-")
        start = int(byte_range[0])
        end   = int(byte_range[1]) if byte_range[1] else file_size - 1
        length = end - start + 1
        with open(video_path, "rb") as f:
            f.seek(start)
            data = f.read(length)
        resp = Response(data, 206, mimetype="video/mp4", direct_passthrough=True)
        resp.headers["Content-Range"]  = f"bytes {start}-{end}/{file_size}"
        resp.headers["Accept-Ranges"]  = "bytes"
        resp.headers["Content-Length"] = str(length)
    else:
        with open(video_path, "rb") as f:
            data = f.read()
        resp = Response(data, 200, mimetype="video/mp4")
        resp.headers["Accept-Ranges"]  = "bytes"
        resp.headers["Content-Length"] = str(file_size)
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp

# ═══════════════════════════════════════════════════════════════════
#  GLOBAL ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════

@app.errorhandler(Exception)
def handle_exception(e):
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


# ███████████████████████████████████████████████████████████████████
#
#  GAME 1  —  DeC4rɹoD€  (full logic from server1.py, zero changes)
#
# ███████████████████████████████████████████████████████████████████

# ── Paths ────────────────────────────────────────────────────────────────
G1_DATA_DIR   = os.path.join(HERE, "data")
G1_KB_PATH    = os.path.join(G1_DATA_DIR, "rag_knowledge_base.json")
G1_SENTS_PATH = os.path.join(G1_DATA_DIR, "sentences.json")
G1_PROG_PATH  = os.path.join(G1_DATA_DIR, "player_progress.json")
G1_GAN_PATH   = os.path.join(G1_DATA_DIR, "numpy_gan_weights.pkl")

# ── Categories ────────────────────────────────────────────────────────────
G1_CATEGORIES = [
    "Philosophy","General Knowledge","Science","History","Geography",
    "Literature","Wisdom","Motivation","Logic and Reasoning","Culture",
    "Technology","Nature","Psychology","Fun Facts","Thirukkural",
]

G1_CATEGORY_BADGES = {
    "Philosophy":          {"id": "philosopher",    "icon": "🧠", "name": "Philosopher"},
    "General Knowledge":   {"id": "polymath",       "icon": "📚", "name": "Polymath"},
    "Science":             {"id": "scientist",      "icon": "🔬", "name": "Scientist"},
    "History":             {"id": "historian",      "icon": "🏛️",  "name": "Historian"},
    "Geography":           {"id": "explorer",       "icon": "🌍", "name": "Explorer"},
    "Literature":          {"id": "wordsmith",      "icon": "✍️",  "name": "Wordsmith"},
    "Wisdom":              {"id": "sage",           "icon": "🌟", "name": "Sage"},
    "Motivation":          {"id": "champion",       "icon": "🏆", "name": "Champion"},
    "Logic and Reasoning": {"id": "logician",       "icon": "⚙️",  "name": "Logician"},
    "Culture":             {"id": "culturalist",    "icon": "🎭", "name": "Culturalist"},
    "Technology":          {"id": "technologist",   "icon": "💻", "name": "Technologist"},
    "Nature":              {"id": "naturalist",     "icon": "🌿", "name": "Naturalist"},
    "Psychology":          {"id": "psychologist",   "icon": "🧬", "name": "Psychologist"},
    "Fun Facts":           {"id": "trivia_master",  "icon": "🎯", "name": "Trivia Master"},
    "Thirukkural":         {"id": "kural_scholar",  "icon": "🕌", "name": "Kural Scholar"},
}

# ── Nexeon config ─────────────────────────────────────────────────────────
G1_NEXEON_COUNT  = 35
G1_NEXEON_SIZE   = 51
G1_UNLOCK_THRESH = 0.60
G1_NEXEONS_PATH  = os.path.join(G1_DATA_DIR, "nexeons.json")

# ── Scoring ───────────────────────────────────────────────────────────────
G1_POINTS_TABLE      = {1: 300, 2: 300, 3: 250, 4: 200, 5: 150}
G1_CHATBOT_COST      = 150
G1_COMPLETION_BONUS  = 500
G1_MAX_ATTEMPTS      = 5
G1_MAX_HINTS         = 3

# ── Load data ─────────────────────────────────────────────────────────────

def _g1_load_json(path: str, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _g1_save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

g1_sentences_db: dict = _g1_load_json(G1_SENTS_PATH, {})
g1_kb:           dict = _g1_load_json(G1_KB_PATH, {})
g1_nexeons_data: list = _g1_load_json(G1_NEXEONS_PATH, [])

# ── GAN corruption ────────────────────────────────────────────────────────

_g1_gan_weights = None
if os.path.exists(G1_GAN_PATH):
    try:
        with open(G1_GAN_PATH, "rb") as f:
            _g1_gan_weights = pickle.load(f)
        print(f"  [Game1] GAN weights loaded ({_g1_gan_weights['vocab_size']} vocab)")
    except Exception as e:
        print(f"  [Game1] GAN weights failed: {e}")

_G1_GLYPH = {
    'a': '@', 'e': '3', 'i': '!', 'o': '0', 's': '$',
    'b': '8', 'g': '9', 'l': '1', 't': '+', 'z': '2',
}

def g1_corrupt(sentence: str) -> str:
    if _g1_gan_weights is not None:
        return _g1_gan_corrupt(sentence)
    return _g1_rule_corrupt(sentence)

def _g1_rule_corrupt(s: str) -> str:
    out = []
    for ch in s:
        r = random.random()
        if r < 0.18 and ch.lower() in _G1_GLYPH:
            g = _G1_GLYPH[ch.lower()]
            out.append(g.upper() if ch.isupper() else g)
        elif r < 0.25:
            out.append(random.choice('@#$%^&*!~<>?'))
        elif r < 0.30 and ch.isalpha():
            out.append(ch.swapcase())
        else:
            out.append(ch)
    return "".join(out)

def _g1_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def _g1_relu(x):
    return np.maximum(0, x)

def _g1_encode_text(text: str, char_to_idx: dict, max_len: int = 50):
    unk = char_to_idx['<UNK>']
    pad = char_to_idx['<PAD>']
    indices = [char_to_idx.get(c, unk) for c in text[:max_len]]
    while len(indices) < max_len:
        indices.append(pad)
    return np.array(indices)

def _g1_generator_forward(char_indices, gen_w: dict) -> float:
    embedded = gen_w['embedding'][char_indices]
    pooled   = np.mean(embedded, axis=0)
    h1       = _g1_relu(pooled @ gen_w['W1'] + gen_w['b1'])
    prob     = _g1_sigmoid(h1 @ gen_w['W2'] + gen_w['b2'])
    return float(prob[0])

def _g1_gan_corrupt(sentence: str) -> str:
    try:
        W           = _g1_gan_weights
        gen_w       = W['generator']
        char_to_idx = W['char_to_idx']
        target_pats = W['target_patterns']
        indices = _g1_encode_text(sentence, char_to_idx)
        prob    = _g1_generator_forward(indices, gen_w)
        chars   = list(sentence)
        for i, ch in enumerate(sentence):
            if ch == ' ':
                continue
            if i == 0 or sentence[i - 1] == ' ':
                if random.random() < 0.7:
                    continue
            if random.random() < prob:
                if ch in target_pats:
                    chars[i] = random.choice(target_pats[ch])
                elif ch.isalpha():
                    if random.random() < 0.4:
                        chars[i] = '_'
        return ''.join(chars)
    except Exception:
        return _g1_rule_corrupt(sentence)

# ── Progress management ───────────────────────────────────────────────────

def _g1_default_progress() -> dict:
    return {
        "score":          0,
        "streak":         0,
        "round_num":      1,
        "history":        [],
        "best_streak":    0,
        "total_wins":     0,
        "total_rounds":   0,
        "last_played":    None,
        "current_nexeon": 1,
        "nexeon_progress": {
            str(i): {
                "wins":      0,
                "played":    0,
                "collected": [],
                "unlocked":  i == 1,
                "completed": False,
            }
            for i in range(1, G1_NEXEON_COUNT + 1)
        },
    }

def g1_load_progress() -> dict:
    p = _g1_load_json(G1_PROG_PATH, None)
    if p is None:
        return _g1_default_progress()
    if "nexeon_progress" not in p:
        fresh = _g1_default_progress()
        fresh["score"]       = p.get("score", 0)
        fresh["streak"]      = p.get("streak", 0)
        fresh["best_streak"] = p.get("best_streak", 0)
        fresh["total_wins"]  = p.get("total_wins", 0)
        fresh["total_rounds"]= p.get("total_rounds", 0)
        return fresh
    for i in range(1, G1_NEXEON_COUNT + 1):
        key = str(i)
        if key not in p["nexeon_progress"]:
            p["nexeon_progress"][key] = {
                "wins": 0, "played": 0, "collected": [],
                "unlocked": i == 1, "completed": False,
            }
    if "current_nexeon" not in p:
        p["current_nexeon"] = 1
    return p

def g1_save_progress(p: dict):
    p["last_played"] = datetime.datetime.now().isoformat()
    _g1_save_json(G1_PROG_PATH, p)

# ── Round state ────────────────────────────────────────────────────────────

class G1RoundState:
    def __init__(self):
        self.sentence:      str  = ""
        self.corrupted:     str  = ""
        self.category:      str  = "all"
        self.attempts:      int  = 0
        self.hints_shown:   int  = 0
        self.chatbot_used:  bool = False
        self.chatbot_turns: int  = 0
        self.active:        bool = False
        self.kb_entry:      dict = {}

    def reset(self):
        self.__init__()

_g1_state = G1RoundState()

# ── Sentence / nexeon selection ───────────────────────────────────────────

def g1_get_nexeon(nexeon_id: int):
    if 1 <= nexeon_id <= len(g1_nexeons_data):
        return g1_nexeons_data[nexeon_id - 1]
    return None

def g1_pick_nexeon_sentence(nexeon_id: int, progress: dict):
    nex = g1_get_nexeon(nexeon_id)
    if not nex:
        return None
    key      = str(nexeon_id)
    np_entry = progress["nexeon_progress"].get(key, {})
    collected = set(np_entry.get("collected", []))
    deferred  = np_entry.get("deferred", [])
    remaining = [s for s in nex["sentences"] if s not in collected and s not in deferred]
    if remaining:
        return random.choice(remaining)
    if deferred:
        return random.choice(deferred)
    np_entry["collected"] = []
    np_entry["deferred"]  = []
    remaining = nex["sentences"]
    return random.choice(remaining)

# ── Hint generation ────────────────────────────────────────────────────────

try:
    sys.path.insert(0, HERE)
    from hint_generator import HintGenerator as _G1HintGen
    _g1_hint_gen = _G1HintGen()
    print("  [Game1] HintGenerator loaded")
except Exception as _hg_err:
    _g1_hint_gen = None
    print(f"  [Game1] HintGenerator not available: {_hg_err}")

def g1_make_hints(sentence: str, kb_entry: dict, attempts: int) -> list:
    hints = []
    words = sentence.split()
    entry = kb_entry
    if attempts >= 2:
        if _g1_hint_gen:
            raw = _g1_hint_gen.get_hint(sentence, 1)
            hints.append({"num": 1, "text": raw})
        else:
            cat = entry.get("category", entry.get("topic", "General"))
            hints.append({"num": 1, "text": f"Category: {cat}"})
    if attempts >= 3:
        if _g1_hint_gen:
            raw = _g1_hint_gen.get_hint(sentence, 2)
            hints.append({"num": 2, "text": raw})
        else:
            wc = entry.get("word_count", len(words))
            hints.append({"num": 2, "text": f"The sentence has {wc} words."})
    if attempts >= 4:
        if _g1_hint_gen:
            raw = _g1_hint_gen.get_hint(sentence, 3)
            hints.append({"num": 3, "text": raw})
        else:
            word_details = entry.get("word_details", {})
            best_word = best_def = None
            for w, info in word_details.items():
                if info and info.get("definition") and len(info["definition"]) > 10:
                    best_word = w
                    best_def  = info["definition"]
                    break
            if best_word:
                hints.append({
                    "num": 3,
                    "text": f'The word <strong>"{best_word}"</strong> means: <em>{best_def}</em>'
                })
            else:
                letters = " – ".join(
                    f"<strong>{w[0].upper()}</strong>"
                    for w in words if w.isalpha()
                )
                hints.append({"num": 3, "text": f"First letter of each word: {letters}."})
    return hints

# ── Answer comparison ──────────────────────────────────────────────────────

def _g1_normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower().strip())

def g1_is_correct(guess: str, sentence: str) -> bool:
    return _g1_normalise(guess) == _g1_normalise(sentence)

# ── Chatbot ────────────────────────────────────────────────────────────────

_g1_chatbot_mod = None
try:
    import chatbot as _g1_chatbot_mod
    print("  [Game1] chatbot.py loaded")
except Exception as e:
    print(f"  [Game1] chatbot.py not loaded ({e}) — fallback mode")

def _g1_simple_chatbot(msg: str) -> str:
    m = msg.lower()
    s = _g1_state.sentence
    words = s.split()
    if "word" in m and ("count" in m or "many" in m or "how" in m):
        return f"The sentence has **{len(words)} words**."
    if "first" in m and "letter" in m:
        return f"The first letter is **{words[0][0].upper()}**."
    if "last" in m and "letter" in m:
        return f"The last letter of the last word is **{words[-1][-1].upper()}**."
    if "topic" in m or "category" in m or "theme" in m:
        t = _g1_state.kb_entry.get("topic", _g1_state.category)
        return f"The topic is **{t}**."
    if "hint" in m:
        hints = g1_make_hints(s, _g1_state.kb_entry, 2)
        if hints:
            return hints[-1]["text"]
    return ("I can help with: word count, first/last letter, topic/theme, "
            "word definitions, synonyms. What would you like to know?")

# ── GAME 1 API ROUTES ──────────────────────────────────────────────────────

@app.route("/game1/api/stats", methods=["GET"])
def g1_api_stats():
    p     = g1_load_progress()
    total = p.get("total_rounds", 0)
    wins  = p.get("total_wins", 0)
    wr    = round(100 * wins / total) if total > 0 else 0
    nexeon_summary = []
    for i in range(1, G1_NEXEON_COUNT + 1):
        key  = str(i)
        np_e = p["nexeon_progress"].get(key, {})
        played   = np_e.get("played", 0)
        nex_wins = np_e.get("wins", 0)
        pct      = round(100 * nex_wins / played) if played > 0 else 0
        nexeon_summary.append({
            "id":        i,
            "unlocked":  np_e.get("unlocked", i == 1),
            "completed": np_e.get("completed", False),
            "wins":      nex_wins,
            "played":    played,
            "pct":       pct,
            "total":     G1_NEXEON_SIZE,
        })
    return jsonify({
        "score":          p.get("score", 0),
        "streak":         p.get("streak", 0),
        "best_streak":    p.get("best_streak", 0),
        "total":          total,
        "wins":           wins,
        "win_rate":       wr,
        "current_nexeon": p.get("current_nexeon", 1),
        "nexeons":        nexeon_summary,
    })

@app.route("/game1/api/nexeons", methods=["GET"])
def g1_api_get_nexeons():
    p = g1_load_progress()
    result = []
    for i in range(1, G1_NEXEON_COUNT + 1):
        key  = str(i)
        np_e = p["nexeon_progress"].get(key, {})
        played   = np_e.get("played", 0)
        nex_wins = np_e.get("wins", 0)
        pct      = round(100 * nex_wins / played) if played > 0 else 0
        nex      = g1_get_nexeon(i) or {}
        result.append({
            "id":        i,
            "name":      f"NEXEON-{i:02d}",
            "unlocked":  np_e.get("unlocked", i == 1),
            "completed": np_e.get("completed", False),
            "wins":      nex_wins,
            "played":    played,
            "pct":       pct,
            "total":     G1_NEXEON_SIZE,
            "categories": nex.get("categories", []),
        })
    return jsonify({"nexeons": result, "current": p.get("current_nexeon", 1)})

@app.route("/game1/api/nexeon/select", methods=["POST"])
def g1_api_nexeon_select():
    data      = request.get_json() or {}
    nexeon_id = int(data.get("nexeon_id", 1))
    p         = g1_load_progress()
    key       = str(nexeon_id)
    np_e      = p["nexeon_progress"].get(key, {})
    if not np_e.get("unlocked", nexeon_id == 1):
        return jsonify({"error": "Nexeon locked — complete a previous nexeon first"}), 403
    p["current_nexeon"] = nexeon_id
    g1_save_progress(p)
    played   = np_e.get("played", 0)
    nex_wins = np_e.get("wins", 0)
    pct      = round(100 * nex_wins / played) if played > 0 else 0
    return jsonify({
        "nexeon_id": nexeon_id,
        "wins":      nex_wins,
        "played":    played,
        "pct":       pct,
        "total":     G1_NEXEON_SIZE,
        "completed": np_e.get("completed", False),
    })

@app.route("/game1/api/new_round", methods=["POST"])
def g1_api_new_round():
    p         = g1_load_progress()
    nexeon_id = p.get("current_nexeon", 1)
    key       = str(nexeon_id)
    np_e      = p["nexeon_progress"].get(key, {})
    if not np_e.get("unlocked", nexeon_id == 1):
        return jsonify({"error": "Nexeon locked — complete previous nexeon first"}), 403
    sentence = g1_pick_nexeon_sentence(nexeon_id, p)
    if not sentence:
        return jsonify({"error": "No sentences available"}), 500
    _g1_state.reset()
    _g1_state.sentence  = sentence
    _g1_state.corrupted = g1_corrupt(sentence)
    _g1_state.category  = f"NEXEON-{nexeon_id:02d}"
    _g1_state.active    = True
    _g1_state.kb_entry  = g1_kb.get(sentence, {})
    np_e = p["nexeon_progress"][key]
    if sentence not in np_e["collected"]:
        np_e["collected"].append(sentence)
    p["total_rounds"] = p.get("total_rounds", 0) + 1
    p["round_num"]    = p["total_rounds"]
    g1_save_progress(p)
    pts = G1_POINTS_TABLE.get(1, 300)
    return jsonify({
        "corrupted":         _g1_state.corrupted,
        "nexeon_id":         nexeon_id,
        "word_count":        len(sentence.split()),
        "category":          _g1_state.category,
        "attempts":          0,
        "hints":             [],
        "points_if_correct": pts,
        "score":             p["score"],
        "streak":            p["streak"],
        "round_num":         p["round_num"],
        "chatbot_used":      False,
    })

@app.route("/game1/api/guess", methods=["POST"])
def g1_api_guess():
    if not _g1_state.active:
        return jsonify({"error": "No active round"}), 400
    data  = request.get_json() or {}
    guess = data.get("guess", "").strip()
    p     = g1_load_progress()
    if g1_is_correct(guess, _g1_state.sentence):
        _g1_state.active = False
        attempts         = _g1_state.attempts + 1
        pts              = G1_POINTS_TABLE.get(attempts, 150)
        nexeon_id        = p.get("current_nexeon", 1)
        key              = str(nexeon_id)
        np_e             = p["nexeon_progress"][key]
        p["score"]       = p.get("score", 0) + pts
        p["streak"]      = p.get("streak", 0) + 1
        p["best_streak"] = max(p.get("best_streak", 0), p["streak"])
        p["total_wins"]  = p.get("total_wins", 0) + 1
        np_e["wins"]   = np_e.get("wins", 0) + 1
        np_e["played"] = np_e.get("played", 0) + 1
        played      = np_e["played"]
        nex_wins    = np_e["wins"]
        win_pct     = nex_wins / played if played > 0 else 0
        nexeon_complete = False
        next_unlocked   = False
        if played >= G1_NEXEON_SIZE and win_pct >= G1_UNLOCK_THRESH and not np_e.get("completed"):
            np_e["completed"] = True
            nexeon_complete   = True
            next_id  = nexeon_id + 1
            next_key = str(next_id)
            if next_id <= G1_NEXEON_COUNT and next_key in p["nexeon_progress"]:
                p["nexeon_progress"][next_key]["unlocked"] = True
                next_unlocked = True
        p["nexeon_progress"][key] = np_e
        p.setdefault("history", []).append({
            "sentence":  _g1_state.sentence,
            "nexeon_id": nexeon_id,
            "won":       True,
            "attempts":  attempts,
            "points":    pts,
            "round":     p.get("round_num", 1),
        })
        g1_save_progress(p)
        return jsonify({
            "result":          "correct",
            "answer":          _g1_state.sentence,
            "points":          pts,
            "streak":          p["streak"],
            "best_streak":     p["best_streak"],
            "attempts":        attempts,
            "score":           p["score"],
            "nexeon_id":       nexeon_id,
            "nexeon_wins":     np_e["wins"],
            "nexeon_played":   np_e["played"],
            "nexeon_pct":      round(100 * np_e["wins"] / np_e["played"]) if np_e["played"] > 0 else 0,
            "nexeon_complete": nexeon_complete,
            "next_unlocked":   next_unlocked,
            "hints":           g1_make_hints(_g1_state.sentence, _g1_state.kb_entry, 5),
        })
    # Wrong guess
    _g1_state.attempts += 1
    attempts_left = G1_MAX_ATTEMPTS - _g1_state.attempts
    hints = g1_make_hints(_g1_state.sentence, _g1_state.kb_entry, _g1_state.attempts)
    pts   = G1_POINTS_TABLE.get(_g1_state.attempts + 1, 150)
    if _g1_state.attempts >= G1_MAX_ATTEMPTS:
        _g1_state.active = False
        p["streak"] = 0
        p.setdefault("history", []).append({
            "sentence": _g1_state.sentence,
            "category": _g1_state.category,
            "won":      False,
            "attempts": G1_MAX_ATTEMPTS,
            "points":   0,
            "round":    p.get("round_num", 1),
        })
        g1_save_progress(p)
        return jsonify({
            "result": "failed",
            "answer": _g1_state.sentence,
            "hints":  hints,
            "score":  p["score"],
            "streak": 0,
        })
    g1_save_progress(p)
    return jsonify({
        "result":            "wrong",
        "attempts":          _g1_state.attempts,
        "attempts_left":     attempts_left,
        "hints":             hints,
        "points_if_correct": pts,
        "score":             p["score"],
    })

@app.route("/game1/api/hint", methods=["POST"])
def g1_api_hint():
    if not _g1_state.active:
        return jsonify({"hint": "No active round"}), 400
    _g1_state.hints_shown = min(_g1_state.hints_shown + 1, G1_MAX_HINTS)
    hints = g1_make_hints(_g1_state.sentence, _g1_state.kb_entry, _g1_state.hints_shown)
    if hints:
        return jsonify({"hint": hints[-1]["text"], "hint_num": _g1_state.hints_shown})
    return jsonify({"hint": "No more hints available.", "hint_num": _g1_state.hints_shown})

@app.route("/game1/api/skip", methods=["POST"])
def g1_api_skip():
    p = g1_load_progress()
    if not _g1_state.active:
        return jsonify({"error": "No active round"}), 400
    nexeon_id    = p.get("current_nexeon", 1)
    key          = str(nexeon_id)
    np_e         = p["nexeon_progress"].get(key, {})
    skipped_sent = _g1_state.sentence
    deferred = np_e.get("deferred", [])
    if skipped_sent not in deferred:
        deferred.append(skipped_sent)
    np_e["deferred"] = deferred
    p["nexeon_progress"][key] = np_e
    g1_save_progress(p)
    next_sentence = g1_pick_nexeon_sentence(nexeon_id, p)
    if not next_sentence:
        return jsonify({"error": "No sentences available"}), 500
    _g1_state.reset()
    _g1_state.sentence  = next_sentence
    _g1_state.corrupted = g1_corrupt(next_sentence)
    _g1_state.category  = f"NEXEON-{nexeon_id:02d}"
    _g1_state.active    = True
    _g1_state.kb_entry  = g1_kb.get(next_sentence, {})
    np_e2 = p["nexeon_progress"][key]
    if next_sentence not in np_e2["collected"]:
        np_e2["collected"].append(next_sentence)
    p["total_rounds"] = p.get("total_rounds", 0) + 1
    p["round_num"]    = p["total_rounds"]
    p["nexeon_progress"][key] = np_e2
    g1_save_progress(p)
    deferred_count = len(np_e2.get("deferred", []))
    pts = G1_POINTS_TABLE.get(1, 300)
    return jsonify({
        "skipped":        True,
        "deferred_count": deferred_count,
        "corrupted":      _g1_state.corrupted,
        "word_count":     len(next_sentence.split()),
        "nexeon_id":      nexeon_id,
        "category":       _g1_state.category,
        "round_num":      p["round_num"],
        "score":          p["score"],
        "streak":         p["streak"],
        "points_if_correct": pts,
    })

@app.route("/game1/api/chatbot/activate", methods=["POST"])
def g1_api_chatbot_activate():
    if not _g1_state.active:
        return jsonify({"error": "No active round"}), 400
    if _g1_state.chatbot_used:
        return jsonify({"error": "Chatbot already used this round"}), 400
    p = g1_load_progress()
    p["score"] = max(0, p.get("score", 0) - G1_CHATBOT_COST)
    g1_save_progress(p)
    _g1_state.chatbot_used  = True
    _g1_state.chatbot_turns = 0
    try:
        from chatbot import RAGChatbot
        app._g1_chatbot = RAGChatbot(kb_path=G1_KB_PATH)
        greeting = app._g1_chatbot.start_session(_g1_state.sentence)
    except Exception as e:
        app._g1_chatbot = None
        greeting = ("Hi! I'm your AI assistant. Ask me about word meanings, "
                    "synonyms, topic hints, or how many words the sentence has.")
    return jsonify({"greeting": greeting, "score": p["score"]})

@app.route("/game1/api/chatbot/chat", methods=["POST"])
def g1_api_chatbot_chat():
    if not _g1_state.chatbot_used:
        return jsonify({"error": "Chatbot not activated"}), 400
    if not _g1_state.active:
        return jsonify({"error": "Round is over"}), 400
    MAX_TURNS = 8
    if _g1_state.chatbot_turns >= MAX_TURNS:
        return jsonify({"response": "Out of queries for this round.", "turns_left": 0})
    data = request.get_json() or {}
    msg  = data.get("message", "").strip()
    if not msg:
        return jsonify({"error": "Empty message"}), 400
    _g1_state.chatbot_turns += 1
    try:
        chatbot = getattr(app, "_g1_chatbot", None)
        if chatbot:
            response   = chatbot.chat(msg)
            turns_left = chatbot.turns_left
        else:
            response   = _g1_simple_chatbot(msg)
            turns_left = MAX_TURNS - _g1_state.chatbot_turns
    except Exception as e:
        response   = f"Sorry, I had trouble with that. ({e})"
        turns_left = MAX_TURNS - _g1_state.chatbot_turns
    return jsonify({"response": response, "turns_left": turns_left})

@app.route("/game1/api/reset_progress", methods=["POST"])
def g1_api_reset_progress():
    p = _g1_default_progress()
    g1_save_progress(p)
    _g1_state.reset()
    return jsonify({"ok": True, "reset": True})

@app.route("/game1/api/save_progress", methods=["POST"])
def g1_api_save_progress():
    data = request.get_json() or {}
    p    = g1_load_progress()
    prof = data.get("profile", {})
    for key in ("score", "streak", "best_streak"):
        if key in prof:
            try:
                p[key] = int(prof[key])
            except (ValueError, TypeError):
                pass
    g1_save_progress(p)
    return jsonify({"ok": True, "saved": True})

@app.route("/game1/api/load_progress", methods=["POST"])
def g1_api_load_progress():
    data = request.get_json() or {}
    p    = g1_load_progress()
    for key in ("score", "streak", "best_streak", "total_rounds"):
        if key in data:
            try:
                p[key] = int(data[key])
            except (ValueError, TypeError):
                pass
    g1_save_progress(p)
    return jsonify({"ok": True, "loaded": True})


# ███████████████████████████████████████████████████████████████████
#
#  GAME 2  —  City Sweep  (full logic from server2.py, zero changes)
#
# ███████████████████████████████████████████████████████████████████

# ── GAN + VAE ─────────────────────────────────────────────────────────────

_g2_gan = None
_g2_vae = None
try:
    from gan import GANInference
    from vae import VAEInference
    _g2_gan = GANInference(
        weights_path=os.path.join(HERE, "models", "gan_weights.pt"),
        rows=10, cols=10, n_ais=10,
    )
    _g2_vae = VAEInference(
        weights_path=os.path.join(HERE, "models", "vae_weights.pt")
    )
    print("  [Game2] GAN + VAE loaded OK")
except Exception as e:
    print(f"  [Game2] GAN/VAE not loaded ({e}) — using random placement")

# ── Game 2 state ──────────────────────────────────────────────────────────

G2_ANOMALY_COST    = 100   # yellow VAE scan cost
G2_VAE_REVEAL_COST = 500   # red VAE reveal cost

G2_DIFFICULTY = {
    "recruit":   {"rows": 6,  "cols": 6,  "ais": 5},
    "operative": {"rows": 8,  "cols": 8,  "ais": 8},
    "elite":     {"rows": 10, "cols": 10, "ais": 10},
    "nightmare": {"rows": 12, "cols": 12, "ais": 14},
}

_g2_state = {
    "city": None, "revealed": [], "flagged": [],
    "score": 1000, "status": "idle",
    "round": 0, "streak": 0,
    "anomaly_used": False, "difficulty": "elite",
    # VAE auto-hint tracking
    "correct_count": 0,       # total safe reveals this game
    "last_auto_batch": 0,     # which multiple-of-5 we last triggered on
    # Paid VAE reveal
    "vae_reveal_used": False,
}

# ── City generation ───────────────────────────────────────────────────────

def _g2_generate_city(diff_name):
    diff      = G2_DIFFICULTY.get(diff_name, G2_DIFFICULTY["elite"])
    target_rows = diff["rows"]
    target_cols = diff["cols"]
    n_ais       = diff["ais"]
    if _g2_gan:
        _g2_gan.rows = 10; _g2_gan.cols = 10; _g2_gan.n_ais = 10
        city_base = _g2_gan.generate_city()
        prob_10x10 = np.array(city_base["probability_map"], dtype=np.float32)
        if target_rows <= 10 and target_cols <= 10:
            prob_2d = prob_10x10[:target_rows, :target_cols].copy()
        else:
            city_b = _g2_gan.generate_city()
            prob_b = np.array(city_b["probability_map"], dtype=np.float32)
            prob_2d = np.zeros((target_rows, target_cols), dtype=np.float32)
            r_fill = min(10, target_rows)
            c_fill = min(10, target_cols)
            prob_2d[:r_fill, :c_fill] = prob_10x10[:r_fill, :c_fill]
            if target_rows > 10:
                prob_2d[10:, :c_fill] = prob_b[:target_rows-10, :c_fill]
            if target_cols > 10:
                prob_2d[:r_fill, 10:] = prob_b[:r_fill, :target_cols-10]
            if target_rows > 10 and target_cols > 10:
                prob_2d[10:, 10:] = prob_b[:target_rows-10, :target_cols-10]
    else:
        prob_2d = np.random.rand(target_rows, target_cols).astype(np.float32)
    flat  = prob_2d.flatten().astype(np.float64)
    flat  = np.clip(flat, 1e-6, None)
    flat /= flat.sum()
    n_cells = target_rows * target_cols
    n_ais   = min(n_ais, n_cells - 1)
    ai_flat = np.random.choice(n_cells, size=n_ais, replace=False, p=flat)
    ai_positions = [
        [int(idx // target_cols), int(idx % target_cols)] for idx in ai_flat
    ]
    return {
        "rows": target_rows, "cols": target_cols,
        "ai_positions": ai_positions,
        "probability_map": prob_2d.tolist(),
    }

def _g2_ai_set():
    return set(tuple(p) for p in _g2_state["city"]["ai_positions"])

def _g2_count_ai_neighbors(r, c, ai_set_):
    return sum(
        1 for dr in [-1, 0, 1] for dc in [-1, 0, 1]
        if (dr or dc) and (r+dr, c+dc) in ai_set_
    )

def _g2_safe_cells():
    city   = _g2_state["city"]
    ai_set_ = _g2_ai_set()
    return [
        (r, c)
        for r in range(city["rows"])
        for c in range(city["cols"])
        if (r, c) not in ai_set_
    ]

def _g2_check_win():
    safe     = set(_g2_safe_cells())
    revealed = set(tuple(x) for x in _g2_state["revealed"])
    return safe.issubset(revealed)

# ── GAME 2 API ROUTES ──────────────────────────────────────────────────────

@app.route("/game2/api/health")
def g2_health():
    return jsonify({
        "status": "ok",
        "models": {
            "gan": os.path.exists(os.path.join(HERE, "models", "gan_weights.pt")),
            "vae": os.path.exists(os.path.join(HERE, "models", "vae_weights.pt")),
        },
    })

@app.route("/game2/api/new_game", methods=["POST"])
def g2_new_game():
    try:
        data = request.get_json(silent=True) or {}
        difficulty = data.get("difficulty", "elite")
        if difficulty not in G2_DIFFICULTY:
            difficulty = "elite"
        city = _g2_generate_city(difficulty)
        _g2_state.update({
            "city": city, "revealed": [], "flagged": [],
            "status": "playing", "round": _g2_state["round"] + 1,
            "anomaly_used": False, "difficulty": difficulty,
            "correct_count": 0, "last_auto_batch": 0,
            "vae_reveal_used": False,
        })
        anomaly_map = _g2_vae.scan_grid(city["probability_map"]) if _g2_vae else \
                      [[0] * city["cols"] for _ in range(city["rows"])]
        return jsonify({
            "rows":       city["rows"],
            "cols":       city["cols"],
            "ai_count":   len(city["ai_positions"]),
            "difficulty": difficulty,
            "round":      _g2_state["round"],
            "score":      _g2_state["score"],
            "anomaly_map": anomaly_map,
            "message":    f"CITY GENERATED — {len(city['ai_positions'])} MINI-AIs EMBEDDED",
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/game2/api/click", methods=["POST"])
def g2_click_cell():
    try:
        if _g2_state["status"] != "playing":
            return jsonify({"error": "No active game"}), 400
        data = request.get_json(silent=True) or {}
        r, c = int(data.get("row", 0)), int(data.get("col", 0))
        city = _g2_state["city"]
        revealed_tuples = set(tuple(x) for x in _g2_state["revealed"])
        if (r, c) in revealed_tuples:
            return jsonify({"result": "already_revealed"})
        ai_set_ = _g2_ai_set()
        if (r, c) in ai_set_:
            _g2_state["status"] = "lost"
            _g2_state["streak"] = 0
            return jsonify({
                "result": "ai_hit", "status": "lost",
                "row": r, "col": c,
                "ai_positions": city["ai_positions"],
                "score": _g2_state["score"], "streak": 0,
                "message": "MINI-AI DETECTED — BREACH! GAME OVER",
            })
        _g2_state["revealed"].append([r, c])
        neighbor_count = _g2_count_ai_neighbors(r, c, ai_set_)
        _g2_state["score"] += 10

        # ── Auto-VAE: fire hint every 5 safe reveals ──
        _g2_state["correct_count"] += 1
        auto_vae = None
        batch_num = _g2_state["correct_count"] // 5
        if batch_num > _g2_state["last_auto_batch"]:
            _g2_state["last_auto_batch"] = batch_num
            # Pick a random UNREVEALED safe cell as the hint
            revealed_so_far = set(tuple(x) for x in _g2_state["revealed"])
            unrevealed_safe_hint = [
                (sr, sc)
                for (sr, sc) in _g2_safe_cells()
                if (sr, sc) not in revealed_so_far
            ]
            if unrevealed_safe_hint:
                hint_cell = random.choice(unrevealed_safe_hint)
                hr, hc = hint_cell
                nb = _g2_count_ai_neighbors(hr, hc, ai_set_)
                auto_vae = {"cell": [hr, hc], "ai_neighbors": nb, "batch": batch_num}

        won = _g2_check_win()
        bonus = 0
        if won:
            _g2_state["status"] = "won"
            _g2_state["streak"] += 1
            bonus = 200 + _g2_state["streak"] * 20
            _g2_state["score"] += bonus
        return jsonify({
            "result":        "safe",
            "status":        _g2_state["status"],
            "row":           r, "col": c,
            "neighbor_count": neighbor_count,
            "points_gained": 10,
            "score":         _g2_state["score"],
            "streak":        _g2_state["streak"],
            "revealed_count": len(_g2_state["revealed"]),
            "safe_total":    len(_g2_safe_cells()),
            "won":           won,
            "bonus":         bonus,
            "auto_vae":      auto_vae,
            "ai_positions":  city["ai_positions"] if won else [],
            "message":       "SAFE NODE" if not won else f"CITY CLEARED! +{bonus} BONUS",
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/game2/api/flag", methods=["POST"])
def g2_flag_cell():
    try:
        if _g2_state["status"] != "playing":
            return jsonify({"error": "No active game"}), 400
        data = request.get_json(silent=True) or {}
        r, c = int(data.get("row", 0)), int(data.get("col", 0))
        flagged_tuples = [tuple(f) for f in _g2_state["flagged"]]
        if (r, c) in flagged_tuples:
            _g2_state["flagged"] = [f for f in _g2_state["flagged"] if tuple(f) != (r, c)]
            action = "unflagged"
        else:
            _g2_state["flagged"].append([r, c])
            action = "flagged"
        return jsonify({"action": action, "row": r, "col": c, "flagged": _g2_state["flagged"]})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/game2/api/anomaly_detect", methods=["POST"])
def g2_anomaly_detect():
    try:
        if _g2_state["status"] != "playing":
            return jsonify({"error": "No active game"}), 400
        if _g2_state["score"] < G2_ANOMALY_COST:
            return jsonify({"error": f"Not enough points. Need {G2_ANOMALY_COST}, have {_g2_state['score']}"}), 400
        if _g2_state["anomaly_used"]:
            return jsonify({"error": "Anomaly detection already used this round"}), 400
        ai_set_ = _g2_ai_set()
        rev_set = set(tuple(x) for x in _g2_state["revealed"])
        city = _g2_state["city"]
        # Pick from UNREVEALED safe cells (not already clicked by player)
        unrevealed_safe = [
            (r, c)
            for r in range(city["rows"])
            for c in range(city["cols"])
            if (r, c) not in ai_set_ and (r, c) not in rev_set
        ]
        if not unrevealed_safe:
            return jsonify({"error": "No unrevealed safe cells left to hint"}), 400
        # Pick the unrevealed safe cell with the most AI neighbours (most useful hint)
        scored = [
            (pos, _g2_count_ai_neighbors(pos[0], pos[1], ai_set_))
            for pos in unrevealed_safe
        ]
        scored.sort(key=lambda x: -x[1])
        best_pos, ai_count = scored[0]
        scan_r, scan_c = best_pos
        _g2_state["score"]       -= G2_ANOMALY_COST
        _g2_state["anomaly_used"] = True
        return jsonify({
            "scan_cell":   [scan_r, scan_c],
            "ai_neighbors": ai_count,
            "cost":        G2_ANOMALY_COST,
            "score":       _g2_state["score"],
            "message":     f"VAE SCAN [{scan_r},{scan_c}] → {ai_count} AI(s) nearby",
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/game2/api/vae_reveal", methods=["POST"])
def g2_vae_reveal():
    """Paid mechanic (once per round, −500 pts): reveals ONE random AI cell position."""
    try:
        if _g2_state["status"] != "playing":
            return jsonify({"error": "No active game"}), 400
        if _g2_state["vae_reveal_used"]:
            return jsonify({"error": "VAE reveal already used this round"}), 400
        if _g2_state["score"] < G2_VAE_REVEAL_COST:
            return jsonify({"error": f"Not enough points. Need {G2_VAE_REVEAL_COST}, have {_g2_state['score']}"}), 400
        city    = _g2_state["city"]
        rev_set = set(tuple(x) for x in _g2_state["revealed"])
        unrevealed_ais = [p for p in city["ai_positions"] if tuple(p) not in rev_set]
        if not unrevealed_ais:
            return jsonify({"error": "No hidden AIs left to reveal"}), 400
        chosen = random.choice(unrevealed_ais)
        _g2_state["vae_reveal_used"] = True
        _g2_state["score"] -= G2_VAE_REVEAL_COST
        return jsonify({
            "row":     chosen[0],
            "col":     chosen[1],
            "cost":    G2_VAE_REVEAL_COST,
            "score":   _g2_state["score"],
            "message": f"VAE REVEAL → AI at [{chosen[0]},{chosen[1]}]",
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/game2/api/stats")
def g2_stats():
    return jsonify({
        "score":  _g2_state["score"],
        "round":  _g2_state["round"],
        "streak": _g2_state["streak"],
        "status": _g2_state["status"],
    })

@app.route("/game2/api/reset", methods=["POST"])
def g2_reset():
    _g2_state.update({
        "score": 1000, "round": 0, "streak": 0, "status": "idle",
        "city": None, "revealed": [], "flagged": [], "anomaly_used": False,
        "correct_count": 0, "last_auto_batch": 0,
        "auto_batch_cells": [], "vae_reveal_used": False,
    })
    return jsonify({"message": "SYSTEM RESET"})


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "data"),   exist_ok=True)
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    print("=" * 60)
    print("  A.I. WARZONE  —  http://localhost:5000")
    print("  /         → Hub (intro + mode selection)")
    print("  /game1    → DeC4rɹoD€ (full nexeon engine)")
    print("  /game2    → City Sweep (GAN city defense)")
    print(f"  GAN :  {'OK' if _g2_gan else 'not loaded — random mode'}")
    print(f"  VAE :  {'OK' if _g2_vae else 'not loaded'}")
    print(f"  G1 KB: {len(g1_kb)} entries | {len(g1_nexeons_data)} nexeons | {sum(len(v) for v in g1_sentences_db.values() if isinstance(v, list))} sentences")
    print(f"  Hint : {'HintGenerator OK' if _g1_hint_gen else 'fallback mode'}")
    print(f"  Bot  : {'chatbot.py OK' if _g1_chatbot_mod else 'fallback mode'}")
    print("=" * 60)
    app.run(debug=True, port=5000)
