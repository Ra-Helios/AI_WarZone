"""
hint_generator.py — Generates 3 progressive hints per round.
Revealed after attempts 2, 3, 4 (configurable in server.py).

Improvement over old version:
  - Hint 1 now uses the ACTUAL category + topic from the RAG knowledge base
    instead of a hardcoded keyword map (which missed History, Geography,
    Literature, Culture, Thirukkural, Fun Facts, Psychology entirely).
  - Hint 2 uses word_details from KB for real synonyms instead of a 24-word
    hand-written synonym dict.
  - Hint 3 unchanged (structural breakdown — first letter of each word).
  - Falls back gracefully if KB is unavailable.
  - No API calls. No pretrained models. Pure Python + JSON.
"""
import os, sys, re, json

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── KB loader (shared with server.py) ────────────────────────────────────
_KB: dict = {}

def _load_kb():
    global _KB
    if _KB:
        return _KB
    candidates = [
        os.path.join(_HERE, "data", "rag_knowledge_base.json"),
        os.path.join(_HERE, "rag_knowledge_base.json"),
        "data/rag_knowledge_base.json",
        "rag_knowledge_base.json",
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                _KB = json.load(f)
            print(f"[HintGen] KB loaded: {len(_KB)} entries from {p}")
            return _KB
    print("[HintGen] WARNING: KB not found — using fallback topic detection")
    return {}


# ── Fallback topic detection (used only if KB unavailable) ───────────────
_CATEGORY_TOPICS = {
    "Philosophy":          ("Philosophy – Ideas & Existence",    "exploring fundamental questions about existence, knowledge, and ethics"),
    "General Knowledge":   ("General – Life & Ideas",            "a reflective statement about life, values, or ideas"),
    "Science":             ("Science – Nature & Universe",       "the systematic study of the natural world"),
    "History":             ("History – Human Journey",           "significant events and figures that shaped civilisation"),
    "Geography":           ("Geography – World & Places",        "the physical and human features of Earth"),
    "Literature":          ("Literature – Language & Story",     "the art of written expression and storytelling"),
    "Wisdom":              ("Wisdom – Insight & Reflection",     "practical insight and timeless lessons about living well"),
    "Motivation":          ("Motivation – Drive & Achievement",  "principles that inspire perseverance and achievement"),
    "Logic and Reasoning": ("Logic – Thinking & Argument",       "the principles of valid reasoning and critical thinking"),
    "Culture":             ("Culture – Society & Expression",    "shared beliefs, arts, and values of human communities"),
    "Technology":          ("Technology – Innovation & Progress","the development and impact of digital innovation"),
    "Nature":              ("Nature – Earth & Living World",     "the natural environment and living organisms of Earth"),
    "Psychology":          ("Psychology – Mind & Behaviour",     "the scientific study of the human mind and behaviour"),
    "Fun Facts":           ("Fun Facts – Surprising Truths",     "surprising and delightful facts about the world"),
    "Thirukkural":         ("Thirukkural – Tamil Wisdom",        "timeless ethical wisdom from the ancient Tamil classic"),
}

def _topic_from_kb(sentence: str) -> tuple[str, str]:
    """Get topic and explanation from KB. Falls back to category map then generic."""
    kb = _load_kb()
    if sentence in kb:
        doc = kb[sentence]
        topic     = doc.get("topic", "")
        topic_exp = doc.get("topic_explanation", "")
        category  = doc.get("category", "")
        if topic and topic_exp:
            return topic, f"This sentence is from the **{category}** category: {topic_exp}."
    # Fallback: scan KB for closest match by first few words
    if kb:
        prefix = " ".join(sentence.split()[:4]).lower()
        for sent, doc in kb.items():
            if sent.lower().startswith(prefix[:15]):
                return doc.get("topic","General"), doc.get("topic_explanation","")
    return "General Knowledge", "This is an interesting sentence worth thinking about carefully."

def _synonyms_from_kb(sentence: str) -> tuple[str, list]:
    """Get a key word + its synonyms from KB word_details."""
    kb = _load_kb()
    doc = kb.get(sentence, {})
    word_details = doc.get("word_details", {})
    for word, info in word_details.items():
        if not info:
            continue
        syns = [s for s in info.get("synonyms", []) if s.lower() != word.lower()]
        if syns:
            return word, syns[:4]
    return "", []


# ── main class ─────────────────────────────────────────────────────────────
class HintGenerator:
    def __init__(self):
        self._cache: dict = {}
        _load_kb()
        print("✅ Hint Generator ready!")

    # ── public API ───────────────────────────────────────────────────────
    def prepare(self, sentence: str) -> list:
        """Generate and cache 3 hints for the sentence. Returns list of 3 strings."""
        if sentence in self._cache:
            return self._cache[sentence]
        hints = [
            self._hint1(sentence),
            self._hint2(sentence),
            self._hint3(sentence),
        ]
        self._cache[sentence] = hints
        return hints

    def generate_all_hints(self, sentence: str) -> list:
        """Alias used by server.py."""
        return self.prepare(sentence)

    def get_hint(self, sentence: str, hint_num: int) -> str:
        """Return a single hint (1-indexed). hint_num: 1, 2, or 3."""
        hints = self.prepare(sentence)
        idx = max(0, min(hint_num - 1, len(hints) - 1))
        return hints[idx]

    # ── hint builders ────────────────────────────────────────────────────
    def _hint1(self, sentence: str) -> str:
        """Hint 1: Category of the sentence — plain text, no markdown."""
        kb  = _load_kb()
        doc = kb.get(sentence, {})
        category = doc.get("category", "")
        topic    = doc.get("topic", "")
        if category:
            if topic and topic != category:
                # Strip the "Category – " prefix from topic if present
                topic_label = topic.split("–")[-1].strip() if "–" in topic else topic
                return f"Category: {category} — {topic_label}"
            return f"Category: {category}"
        topic, _ = _topic_from_kb(sentence)
        return f"Category: {topic}"

    def _hint2(self, sentence: str) -> str:
        """Hint 2: First & last letter of the longest word — plain text."""
        words = [re.sub(r'[^a-zA-Z]', '', w) for w in sentence.split()]
        words = [w for w in words if w]
        if not words:
            return "No structural hint available."
        longest = max(words, key=len)
        first_l = longest[0].upper()
        last_l  = longest[-1].upper()
        length  = len(longest)
        return (
            f"The longest word has {length} letters. "
            f"It starts with '{first_l}' and ends with '{last_l}'."
        )

    def _hint3(self, sentence: str) -> str:
        """Hint 3: Meaning of the sentence — plain text, sentence words scrubbed."""
        kb  = _load_kb()
        doc = kb.get(sentence, {})
        # Use topic_explanation (simpler) not sentence_explanation (too verbose)
        topic_exp = doc.get("topic_explanation", "")
        category  = doc.get("category", "")
        if topic_exp:
            # Scrub sentence words longer than 3 chars
            result = topic_exp[:200]
            for w in sentence.split():
                clean = re.sub(r'[^a-zA-Z]', '', w)
                if len(clean) > 3:
                    result = re.sub(
                        rf'\b{re.escape(clean)}\b', '___',
                        result, flags=re.IGNORECASE
                    )
            prefix = f"This is a {category} sentence. " if category else ""
            return f"{prefix}It is about: {result}{'…' if len(topic_exp) > 200 else ''}"
        return "Think carefully about what each word means as a whole thought."


# ── factory ────────────────────────────────────────────────────────────────
def create_hint_generator() -> HintGenerator:
    return HintGenerator()


# ── test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gen = HintGenerator()
    kb  = _load_kb()
    # Pick one sentence from each category
    from collections import defaultdict
    by_cat = defaultdict(list)
    for s, d in kb.items():
        by_cat[d.get("category","?")].append(s)

    for cat, sents in sorted(by_cat.items()):
        s = sents[0]
        print(f"\n[{cat}] {s}")
        for i, h in enumerate(gen.prepare(s), 1):
            print(f"  Hint {i}: {h}")
