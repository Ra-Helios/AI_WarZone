"""
build_rag_kb.py  —  builds rag_knowledge_base.json from sentences.json
New structure: sentences.json uses { "CategoryName": [...] } instead of difficulty keys.

Run once:  python build_rag_kb.py
"""
from __future__ import annotations
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from word_dictionary import WORD_DICT

# ── Category → topic/explanation mapping ────────────────────────────────

CATEGORY_META = {
    "Philosophy":          ("Philosophy – Ideas & Existence",      "exploring fundamental questions about existence, knowledge, truth, and ethics"),
    "General Knowledge":   ("General – Life & Ideas",              "a reflective statement about life, values, or ideas worth thinking about"),
    "Science":             ("Science – Nature & Universe",         "the systematic study of the natural world through observation and experiment"),
    "History":             ("History – Human Journey",             "significant events, figures, and movements that shaped human civilisation"),
    "Geography":           ("Geography – World & Places",          "the physical and human features of Earth's lands, seas, and regions"),
    "Literature":          ("Literature – Language & Story",       "the art of written expression, storytelling, and linguistic creativity"),
    "Wisdom":              ("Wisdom – Insight & Reflection",       "practical insight and timeless lessons about how to live well"),
    "Motivation":          ("Motivation – Drive & Achievement",    "principles that inspire action, perseverance, and personal achievement"),
    "Logic and Reasoning": ("Logic – Thinking & Argument",        "the principles of valid reasoning, critical thinking, and sound argumentation"),
    "Culture":             ("Culture – Society & Expression",      "the shared beliefs, practices, arts, and values of human communities"),
    "Technology":          ("Technology – Innovation & Progress",  "the development and impact of tools, systems, and digital innovation"),
    "Nature":              ("Nature – Earth & Living World",       "the natural environment, ecosystems, and the living organisms of Earth"),
    "Psychology":          ("Psychology – Mind & Behaviour",       "the scientific study of the human mind, emotions, and behaviour"),
    "Fun Facts":           ("Fun Facts – Surprising Truths",       "surprising, curious, and delightful facts about the world around us"),
    "Thirukkural":         ("Thirukkural – Tamil Wisdom",          "timeless ethical and philosophical wisdom from the ancient Tamil classic"),
}

# ── Stop-words (excluded from keyword extraction) ───────────────────────

STOP = {
    "is","are","was","the","a","an","in","on","at","of","to","for","and","or","but",
    "it","its","this","that","as","not","all","no","be","by","do","so","up","we","you",
    "he","she","they","my","your","will","can","take","from","with","while","ultimately",
    "despite","through","when","over","long","periods","effort","moments","uncertainty",
    "guided","reflection","embraced","fully","experience","matures","accumulated",
    "lessons","accumulate","silent","persistence","practiced","discipline","doubt",
    "produces","reveals","builds","brings","begins","creates","create","form","forms",
    "inspires","leads","unfolds","demands","gives","defines","strengthens","shapes",
    "influences","deepens","every","more","most","less","each","some","have","what",
    "just","also","only","even","than","then","into","upon","does","never","those",
    "these","them","their","although","because","since","which","requires","fear",
    "has","had","been","were","would","could","should","may","might","shall","must",
    "its","our","their","us","me","him","her","who","how","very","too","much","many",
    "about","after","before","between","under","again","further","once","here","there",
    "where","why","both","few","more","most","other","same","own","such","then","than",
    "too","very","just","because","if","while","though","although","however",
}

# ── Word detail extraction ────────────────────────────────────────────────

def extract_word_details(sentence: str) -> dict:
    details = {}
    for raw in sentence.split():
        w = raw.rstrip(".,!?'\"").lower()
        if w in STOP or len(w) < 3:
            continue
        for candidate in [w, w.rstrip("s"), w.rstrip("es"), w.rstrip("ed"),
                          w.rstrip("ing"), w.rstrip("ly"), w.rstrip("er")]:
            if candidate in WORD_DICT:
                entry = WORD_DICT[candidate]
                clean_key = raw.rstrip(".,!?'\"")
                details[clean_key] = {
                    "definition": entry.get("definition", ""),
                    "pos":        entry.get("pos", ""),
                    "synonyms":   entry.get("synonyms", [])[:6],
                    "antonyms":   entry.get("antonyms", [])[:4],
                    "example":    "",  # suppressed — leaks the word
                }
                break
        else:
            clean_key = raw.rstrip(".,!?'\"")
            details[clean_key] = {}
    return details

# ── Keyword extraction ────────────────────────────────────────────────────

def extract_keywords(sentence: str) -> list[str]:
    words = [w.rstrip(".,!?'\"") for w in sentence.split()]
    keywords = []
    seen = set()
    for w in words:
        wl = w.lower()
        if wl not in STOP and len(wl) > 3 and wl not in seen:
            keywords.append(w)
            seen.add(wl)
        if len(keywords) >= 7:
            break
    return keywords

# ── Sentence explanation generator ───────────────────────────────────────

def generate_explanation(sentence: str, category: str, topic: str, topic_exp: str) -> str:
    words = sentence.split()
    wl    = [w.lower().rstrip(".,!?\"'") for w in words]
    n     = len(words)

    # Find the most meaningful word with a dictionary entry
    primary_word = None
    primary_def  = None
    for w in wl:
        if w in STOP or len(w) < 4:
            continue
        for candidate in [w, w.rstrip("s"), w.rstrip("ed"), w.rstrip("ing"), w.rstrip("ly")]:
            if candidate in WORD_DICT:
                primary_word = candidate
                primary_def  = WORD_DICT[candidate]["definition"].rstrip(".")
                break
        if primary_word:
            break

    cat_label = topic.split("–")[-1].strip()

    if primary_word and primary_def:
        return (
            f"This sentence belongs to the {category} category. "
            f"It explores the theme of {cat_label.lower()}: {topic_exp}. "
            f"The word '{primary_word}' means: {primary_def}. "
            f"Think about what the sentence is saying as a whole — each word is a clue."
        )

    return (
        f"This sentence belongs to the {category} category, "
        f"touching on the theme of {cat_label.lower()}. "
        f"It conveys that {topic_exp}. "
        f"Read the corrupted version carefully — the structure and length are preserved."
    )

# ── Main build ────────────────────────────────────────────────────────────

def build(sentences_path: str | None = None, output_path: str | None = None):
    # Locate sentences.json
    if sentences_path is None:
        for rel in ["sentences.json", "data/sentences.json", "../data/sentences.json"]:
            p = os.path.join(HERE, rel)
            if os.path.exists(p):
                sentences_path = p
                break
    if not sentences_path or not os.path.exists(sentences_path):
        raise FileNotFoundError("sentences.json not found — place it in the same folder.")

    with open(sentences_path, encoding="utf-8") as f:
        raw = json.load(f)

    # Output path
    if output_path is None:
        out_dir = os.path.join(HERE, "data")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "rag_knowledge_base.json")

    kb: dict  = {}
    seen: set = set()
    total = with_defn = skipped = 0

    for category, sentences in raw.items():
        # Skip old-format difficulty keys if they slip through
        if category in ("easy", "medium", "hard", "all"):
            continue

        topic, topic_exp = CATEGORY_META.get(
            category,
            (f"{category} – General", f"themes and ideas related to {category.lower()}")
        )

        for sentence in sentences:
            if not isinstance(sentence, str) or not sentence.strip():
                skipped += 1
                continue
            if sentence in seen:
                skipped += 1
                continue
            seen.add(sentence)
            total += 1

            words        = sentence.split()
            word_details = extract_word_details(sentence)
            synonyms_map = {w: info["synonyms"]
                            for w, info in word_details.items()
                            if info.get("synonyms")}
            explanation  = generate_explanation(sentence, category, topic, topic_exp)
            first_letter = words[0][0].upper()  if words else "?"
            last_letter  = words[-1][-1].upper() if words else "?"
            # Strip trailing punctuation from last letter check
            last_word = words[-1].rstrip(".,!?\"'") if words else ""
            last_letter = last_word[-1].upper() if last_word else "?"

            if any(v for v in word_details.values()):
                with_defn += 1

            kb[sentence] = {
                "sentence":             sentence,
                "category":             category,
                "topic":                topic,
                "topic_explanation":    topic_exp,
                "sentence_explanation": explanation,
                "word_count":           len(words),
                "first_letter":         first_letter,
                "last_letter":          last_letter,
                "keywords":             extract_keywords(sentence),
                "word_details":         word_details,
                "sentence_synonyms":    synonyms_map,
            }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

    size_kb = os.path.getsize(output_path) / 1024
    pct     = 100 * with_defn // total if total else 0
    print(f"✅  RAG Knowledge Base rebuilt!")
    print(f"    Entries        : {total}")
    print(f"    With word defs : {with_defn} ({pct}%)")
    print(f"    Skipped        : {skipped}")
    print(f"    Output         : {output_path}")
    print(f"    File size      : {size_kb:.0f} KB")
    return output_path


if __name__ == "__main__":
    build()
