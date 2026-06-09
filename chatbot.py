"""
chatbot.py — RAG-based chatbot for the sentence guessing game.
No external API. Uses rag_knowledge_base.json for retrieval.

ANTI-LEAK RULES (enforced throughout):
  • Never print the actual word name in definition responses
  • Never include the example sentence (it contains the word)
  • _scrub() masks any sentence word that sneaks into a response
"""
import json, re, os, sys as _sys

# Live word dictionary — used as runtime fallback when KB entry is empty
def _load_word_dict():
    candidates = [
        os.path.join(os.path.dirname(__file__), "word_dictionary.py"),
        os.path.join(os.path.dirname(__file__), "../word_dictionary.py"),
        "word_dictionary.py",
    ]
    for p in candidates:
        if os.path.exists(p):
            ns = {}
            exec(open(p).read(), ns)
            return ns.get("WORD_DICT", {})
    return {}

_WORD_DICT = _load_word_dict()

MAX_TURNS = 8
KB_PATH   = os.path.join(os.path.dirname(__file__), "../data/rag_knowledge_base.json")

ORDINALS = {
    "first":1,"second":2,"third":3,"fourth":4,"fifth":5,
    "sixth":6,"seventh":7,"eighth":8,"ninth":9,"tenth":10,
    "eleventh":11,"twelfth":12,"thirteenth":13,"fourteenth":14,"fifteenth":15,
    "sixteenth":16,"seventeenth":17,"eighteenth":18,"nineteenth":19,"twentieth":20,
    "last":-1,
    "1st":1,"2nd":2,"3rd":3,"4th":4,"5th":5,
    "6th":6,"7th":7,"8th":8,"9th":9,"10th":10,
    "11th":11,"12th":12,"13th":13,"14th":14,"15th":15,
    "16th":16,"17th":17,"18th":18,"19th":19,"20th":20,
}

def _ord_to_int(token: str):
    t = token.lower().strip(".,?!")
    if t in ORDINALS:
        return ORDINALS[t]
    # Handle "12th", "21st", "22nd", "33rd" etc.
    m = re.match(r'^(\d+)(?:st|nd|rd|th)$', t)
    if m:
        return int(m.group(1))
    try:
        return int(t)
    except ValueError:
        return None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

class TFIDFRetriever:
    def __init__(self, docs: dict):
        self.keys = list(docs.keys())
        self.docs = docs
        self.vec  = None
        if _SKLEARN and self.keys:
            texts = [
                d.get("sentence_explanation","") + " " + " ".join(d.get("keywords",[]))
                for d in docs.values()
            ]
            self.vec = TfidfVectorizer(stop_words="english")
            self.mat = self.vec.fit_transform(texts)

    def retrieve(self, query: str, k=3):
        if self.vec is None:
            return []
        qv   = self.vec.transform([query])
        sims = cosine_similarity(qv, self.mat)[0]
        idxs = sims.argsort()[::-1][:k]
        return [self.docs[self.keys[i]] for i in idxs if sims[i] > 0]

def _synonym(word: str, doc: dict) -> list:
    """Look up synonyms: KB sentence_synonyms → KB word_details → live WORD_DICT."""
    clean = re.sub(r'[^a-zA-Z]', '', word.lower())
    # 1. sentence_synonyms (fastest)
    for k, v in (doc.get("sentence_synonyms") or {}).items():
        if k.lower() == clean:
            return v if isinstance(v, list) else [str(v)]
    # 2. KB word_details
    for k, v in (doc.get("word_details") or {}).items():
        if k.lower() == clean and v:
            s = v.get("synonyms", [])
            if s:
                return s if isinstance(s, list) else []
    # 3. Live WORD_DICT fallback (handles words not in KB)
    for stem in [clean, clean.rstrip("s"), clean.rstrip("ed"),
                 clean.rstrip("ing"), clean.rstrip("ly"), clean.rstrip("er")]:
        if stem in _WORD_DICT:
            s = _WORD_DICT[stem].get("synonyms", [])
            if s:
                return [x for x in s if x.lower() != clean]
    return []


def _word_info(word: str, doc: dict) -> dict:
    """Look up full word info: KB word_details → live WORD_DICT."""
    clean = re.sub(r'[^a-zA-Z]', '', word.lower())
    # KB first
    for k, v in (doc.get("word_details") or {}).items():
        if k.lower() == clean and v and v.get("definition"):
            return v
    # WORD_DICT fallback with stemming
    for stem in [clean, clean.rstrip("s"), clean.rstrip("ed"),
                 clean.rstrip("ing"), clean.rstrip("ly"), clean.rstrip("er")]:
        if stem in _WORD_DICT:
            entry = _WORD_DICT[stem]
            if entry.get("definition"):
                return entry
    return {}

def _detect_topic(sentence: str, doc: dict) -> str:
    return doc.get("topic", "General")


class RAGChatbot:
    def __init__(self, kb_path: str = KB_PATH):
        self.kb: dict = {}
        self._load_kb(kb_path)
        self.retriever        = TFIDFRetriever(self.kb)
        self.is_active        = False
        self.current_doc      : dict = {}
        self.current_sentence : str  = ""
        self.turns            = 0
        self.turns_left       = MAX_TURNS
        self.max_turns        = MAX_TURNS
        self._hint_idx        = 0

    @property
    def active(self) -> bool:
        return self.is_active

    def _load_kb(self, path: str):
        candidates = [
            path,
            os.path.join(os.path.dirname(__file__), "rag_knowledge_base.json"),
            "data/rag_knowledge_base.json",
            "game_logic/rag_knowledge_base.json",
            "rag_knowledge_base.json",
        ]
        for p in candidates:
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    self.kb = json.load(f)
                print(f"[RAG] Loaded {len(self.kb)} KB entries from {p}")
                return
        print("[RAG] WARNING: knowledge base not found.")

    # ── session ────────────────────────────────────────────────────────────
    def start_session(self, sentence: str) -> str:
        self.current_sentence = sentence
        self.current_doc      = self.kb.get(sentence, {})
        self.is_active        = True
        self.turns            = 0
        self.turns_left       = self.max_turns
        self._hint_idx        = 0

        words    = sentence.split()
        n        = len(words)
        category = self.current_doc.get("category", "")
        wd_count = sum(1 for v in (self.current_doc.get("word_details") or {}).values() if v and v.get("definition"))
        return (
            f"**AI Assistant** — {self.turns_left} queries left\n\n"
            f"**{n} word{'s' if n != 1 else ''}**"
            + (f"  ·  {category}" if category else "")
            + f"  ·  {wd_count} word def{'s' if wd_count != 1 else ''}\n\n"
            f"**Commands:**\n"
            f"• `hint` — genre & category clue\n"
            f"• `Nth word meaning` — definition + POS\n"
            f"• `Nth word synonym` — synonyms only\n"
            f"• `Nth word antonym` — antonyms only\n"
            f"• `Nth word example` — example sentence\n"
            f"• `word count` — length & first/last letters\n"
            f"• `which words` — positions with definitions"
        )

    def end_session(self):
        self.is_active = False

    # ── scrubber ───────────────────────────────────────────────────────────
    def _scrub(self, text: str) -> str:
        """Replace sentence words (>3 chars) in text with [?] to prevent leaking."""
        result = text
        for w in self.current_sentence.split():
            clean = re.sub(r'[^a-zA-Z]', '', w)
            if len(clean) > 3:
                result = re.sub(
                    rf'\b{re.escape(clean)}\b', '[?]',
                    result, flags=re.IGNORECASE
                )
        return result

    # ── main entry ─────────────────────────────────────────────────────────
    def chat(self, user_message: str) -> str:
        if not self.is_active:
            return "Chatbot is not active this round."
        if self.turns_left <= 0:
            return "You've used all your queries for this round."

        self.turns      += 1
        self.turns_left -= 1

        msg   = user_message.strip()
        lower = msg.lower()
        doc   = self.current_doc
        words = self.current_sentence.split()

        # 1. Block answer leaks
        if any(p in lower for p in ["tell me the answer","what is the answer",
                                     "what's the answer","give me the answer",
                                     "just tell me","reveal the"]):
            return ("I can't reveal the answer — that's the challenge! 😄 "
                    "I can help with word meanings, synonyms, or context clues.")

        # 2a. Nth-word synonym (must check BEFORE generic nth-word)
        nth_syn = self._parse_nth_word_synonym(lower)
        if nth_syn is not None:
            return self._resp_nth_word_synonyms(nth_syn, words, doc)

        # 2b. Nth-word definition
        nth = self._parse_nth_word(lower)
        if nth is not None:
            return self._resp_nth_word(nth, words, doc)

        # 2c. Nth-word POS
        nth_pos = self._parse_nth_word_pos(lower)
        if nth_pos is not None:
            return self._resp_nth_word_pos(nth_pos, words, doc)

        # 2d. Nth-word antonym
        nth_ant = self._parse_nth_word_antonym(lower)
        if nth_ant is not None:
            return self._resp_nth_word_antonym(nth_ant, words, doc)

        # 2e. Nth-word example
        nth_ex = self._parse_nth_word_example(lower)
        if nth_ex is not None:
            return self._resp_nth_word_example(nth_ex, words, doc)

        # 3. What does X mean / define X
        word_match = self._parse_word_query(lower)
        if word_match:
            return self._resp_word_def(word_match, doc)

        # 4. Synonyms for X
        syn_match = self._parse_synonym_query(lower)
        if syn_match:
            return self._resp_synonyms(syn_match, doc)

        # 5. List positions with definitions
        if re.search(r'\b(which words|what words|list words|definitions available|'
                     r'words you know|have definitions for)\b', lower):
            return self._resp_list_words(doc)

        # 6. Topic
        if any(p in lower for p in ["topic","category","subject","about","field","domain"]):
            return self._resp_topic(doc)

        # 7. Sentence explanation
        if any(p in lower for p in ["explain","sentence mean","meaning of the sentence",
                                     "what is this sentence","what does the sentence"]):
            return self._resp_explain(doc)

        # 8. Word count
        if any(p in lower for p in ["how many words","word count","length","how long"]):
            return self._resp_wordcount(words)

        # 9. First / last letter — only if NOT asking for meaning/definition/synonyms
        _word_intent = any(p in lower for p in ["meaning","definition","define","synonym","synonyms"])
        if ("first word" in lower or "first letter" in lower) and not _word_intent:
            return f"The first word starts with **'{words[0][0].upper()}'**."
        if ("last word" in lower or "last letter" in lower) and not _word_intent:
            return f"The last word starts with **'{words[-1][0].upper()}'**."
        # "last word meaning/definition" → treat as nth word for last position
        if ("last word" in lower or "last word" in lower) and _word_intent:
            return self._resp_nth_word(len(words), words, doc)
        if ("first word" in lower) and _word_intent:
            return self._resp_nth_word(1, words, doc)

        # 10. Generic hint
        if any(p in lower for p in ["hint","clue","help","stuck","guide","struggling"]):
            return self._resp_rotating_hint(words, doc)

        # 11. RAG fallback
        return self._resp_rag_fallback(msg, doc, words)

    # ── intent parsers ─────────────────────────────────────────────────────
    def _parse_nth_word(self, lower: str):
        patterns = [
            r'(\w+)\s+word\s+(?:definition|meaning|mean|synonym|synonyms)',
            r'define\s+(?:the\s+)?(\w+)\s+word',
            r'(?:what does|meaning of)\s+(?:the\s+)?(\w+)\s+word',
            r'(\w+)\s+word\b',
            r'word\s+(\w+)',
        ]
        for pat in patterns:
            m = re.search(pat, lower)
            if m:
                n = _ord_to_int(m.group(1))
                if n is not None:
                    return n
        return None

    def _parse_nth_word_synonym(self, lower: str):
        """Detect 'Nth word synonym(s)' queries specifically."""
        m = re.search(r'(\w+)\s+word\s+synonym', lower)
        if m:
            n = _ord_to_int(m.group(1))
            if n is not None:
                return n
        return None

    def _parse_nth_word_pos(self, lower: str):
        """Detect 'Nth word pos' or 'Nth word part of speech' queries."""
        m = re.search(r'(\w+)\s+word\s+(?:pos|part\s+of\s+speech|type|class)', lower)
        if m:
            n = _ord_to_int(m.group(1))
            if n is not None:
                return n
        return None

    def _parse_nth_word_antonym(self, lower: str):
        """Detect 'Nth word antonym' queries."""
        m = re.search(r'(\w+)\s+word\s+antonym', lower)
        if m:
            n = _ord_to_int(m.group(1))
            if n is not None:
                return n
        return None

    def _parse_nth_word_example(self, lower: str):
        """Detect 'Nth word example' or 'Nth word in sentence' queries."""
        m = re.search(r'(\w+)\s+word\s+(?:example|use|usage|in\s+sentence|sentence)', lower)
        if m:
            n = _ord_to_int(m.group(1))
            if n is not None:
                return n
        return None

    def _parse_word_query(self, lower: str):
        skip = {"the","a","an","is","in","of","to","for","it","this","that",
                "word","sentence","hint","me","my","i","what","does"}
        for pat in [
            r'what does\s+["\']?(\w+)["\']?\s+mean',
            r'define\s+["\']?(\w+)["\']?$',
            r'meaning of\s+["\']?(\w+)["\']?',
            r'definition of\s+["\']?(\w+)["\']?',
            r'explain\s+["\']?(\w+)["\']?$',
        ]:
            m = re.search(pat, lower)
            if m:
                w = m.group(1).strip()
                if w not in skip:
                    return w
        return None

    def _parse_synonym_query(self, lower: str):
        for pat in [
            r'synonym(?:s)?\s+(?:of|for)\s+["\']?(\w+)["\']?',
            r'another word for\s+["\']?(\w+)["\']?',
            r'similar (?:word|words) (?:to|for)\s+["\']?(\w+)["\']?',
            r'alternative(?:s)?\s+(?:for|to)\s+["\']?(\w+)["\']?',
        ]:
            m = re.search(pat, lower)
            if m:
                return m.group(1).strip()
        return None

    # ── response builders ──────────────────────────────────────────────────

    def _resp_nth_word(self, n: int, words: list, doc: dict) -> str:
        if n == -1:
            n = len(words)
        if n < 1 or n > len(words):
            return (f"The sentence has **{len(words)} words**, "
                    f"so there's no word #{n}. "
                    f"Try a number between 1 and {len(words)}.")

        target = re.sub(r'[^a-zA-Z]', '', words[n-1])
        suf    = {1:"st",2:"nd",3:"rd"}.get(n if n < 20 else n % 10, "th")
        label  = f"{n}{suf}"

        wd = _word_info(target, doc)

        if wd and wd.get("definition"):
            pos  = wd.get("pos", "")
            defn = self._scrub(wd["definition"][:200])
            out  = f"**{label} word**"
            if pos:
                out += f" ({pos})"
            out += f" — {defn}"
            return out

        syns = [s for s in _synonym(target, doc) if s.lower() != target.lower()]
        if syns:
            return f"**{label} word** — synonyms: {', '.join(syns[:5])}"

        word_len = len(target)
        first_ch = target[0].upper() if target else "?"
        return f"**{label} word** — no definition found ({word_len} letters, starts with '{first_ch}')."

    def _resp_word_def(self, word: str, doc: dict) -> str:
        wd = _word_info(word, doc)

        if wd and wd.get("definition"):
            pos  = wd.get("pos", "")
            # ❌ Don't say the word name — it reveals the answer
            # ❌ Don't include example — it contains the word
            defn = self._scrub(wd["definition"][:180])
            syns = [s for s in wd.get("synonyms", [])[:5]
                    if s.lower() != word.lower()]
            out  = f"This word ({pos}): {defn}"
            if syns:
                out += f"\n📖 Synonyms: {', '.join(syns)}"
            return out

        syns = [s for s in _synonym(word, doc) if s.lower() != word.lower()]
        if syns:
            return f"Synonyms: **{', '.join(syns[:4])}**."

        return ("I don't have a definition for that word. "
                "Try asking by position, e.g. '2nd word definition'.")

    def _resp_synonyms(self, word: str, doc: dict) -> str:
        syns = [s for s in _synonym(word, doc) if s.lower() != word.lower()]
        if not syns:
            for k, v in (doc.get("word_details") or {}).items():
                if k.lower() == word.lower() and v:
                    syns = [s for s in v.get("synonyms", [])[:6]
                            if s.lower() != word.lower()]
                    break
        if syns:
            # ❌ Don't say "Synonyms for Compassion:" — reveals the word
            return f"Synonyms: {', '.join(syns[:6])}."
        return "No synonyms available. Try asking by position, e.g. '3rd word definition'."

    def _resp_list_words(self, doc: dict) -> str:
        avail = list((doc.get("word_details") or {}).keys())
        if avail:
            words = self.current_sentence.split()
            positions = []
            for i, w in enumerate(words, 1):
                clean = re.sub(r'[^a-zA-Z]', '', w)
                for k in avail:
                    if k.lower() == clean.lower():
                        suf = {1:"st",2:"nd",3:"rd"}.get(i if i < 20 else i % 10, "th")
                        positions.append(f"{i}{suf}")
                        break
            pos_str = ", ".join(positions) if positions else "several"
            first   = positions[0] if positions else "1st"
            # ❌ Don't list the actual word names
            return (f"I have definitions for words at position(s): **{pos_str}**. "
                    f"Ask me e.g. '{first} word definition'.")
        return "I don't have word definitions for this sentence, but I can give topic hints!"

    def _resp_topic(self, doc: dict) -> str:
        topic   = doc.get("topic", "General")
        explain = doc.get("topic_explanation", "")
        out = f"Topic: **{topic}**."
        if explain:
            out += f" {self._scrub(explain[:160])}"
        return out

    def _resp_explain(self, doc: dict) -> str:
        expl = doc.get("sentence_explanation", "")
        if expl:
            # Scrub all sentence words from the explanation
            return f"💡 {self._scrub(expl[:220])}{'…' if len(expl) > 220 else ''}"
        return "I don't have a detailed explanation, but I can help with hints!"

    def _resp_wordcount(self, words: list) -> str:
        n = len(words)
        return (f"The sentence has **{n} word{'s' if n != 1 else ''}**. "
                f"First letter: **'{words[0][0].upper()}'**, "
                f"last letter: **'{words[-1][0].upper()}'**.")

    def _resp_rotating_hint(self, words: list, doc: dict) -> str:
        """'hint' keyword — returns genre/category with explanation only."""
        category = doc.get("category", "General")
        topic    = doc.get("topic", "")
        explain  = self._scrub(doc.get("topic_explanation", "")[:200])
        out = f"**Genre: {category}**"
        if topic and topic != category:
            out += f" — {topic}"
        if explain:
            out += f"\n{explain}"
        return out

    def _resp_nth_word_synonyms(self, n: int, words: list, doc: dict) -> str:
        if n == -1:
            n = len(words)
        if n < 1 or n > len(words):
            return (f"The sentence has **{len(words)} words**, "
                    f"so there's no word #{n}.")
        target = re.sub(r'[^a-zA-Z]', '', words[n-1])
        suf    = {1:"st",2:"nd",3:"rd"}.get(n if n < 20 else n % 10, "th")
        label  = f"{n}{suf}"

        wd   = _word_info(target, doc)
        syns = [s for s in _synonym(target, doc) if s.lower() != target.lower()]
        if not syns and wd:
            syns = [s for s in wd.get("synonyms", [])[:6] if s.lower() != target.lower()]

        if syns:
            return f"**{label} word** — synonyms: {', '.join(syns[:6])}"
        return f"**{label} word** — no synonyms found."


    def _resp_nth_word_pos(self, n: int, words: list, doc: dict) -> str:
        """Return just the part of speech for the Nth word."""
        if n == -1:
            n = len(words)
        if n < 1 or n > len(words):
            return f"No word #{n} — sentence has {len(words)} words."
        target = re.sub(r'[^a-zA-Z]', '', words[n-1])
        suf    = {1:"st",2:"nd",3:"rd"}.get(n if n < 20 else n % 10, "th")
        label  = f"{n}{suf}"
        wd = _word_info(target, doc)
        pos = wd.get("pos", "") if wd else ""
        if pos:
            return f"**{label} word** — part of speech: {pos}"
        return f"**{label} word** — part of speech not available."

    def _resp_nth_word_antonym(self, n: int, words: list, doc: dict) -> str:
        """Return antonyms for the Nth word."""
        if n == -1:
            n = len(words)
        if n < 1 or n > len(words):
            return f"No word #{n} — sentence has {len(words)} words."
        target = re.sub(r'[^a-zA-Z]', '', words[n-1])
        suf    = {1:"st",2:"nd",3:"rd"}.get(n if n < 20 else n % 10, "th")
        label  = f"{n}{suf}"
        wd   = _word_info(target, doc)
        ants = [a for a in (wd.get("antonyms", []) if wd else [])[:6] if a.lower() != target.lower()]
        if ants:
            return f"**{label} word** — antonyms: {', '.join(ants)}"
        return f"**{label} word** — no antonyms found."

    def _resp_nth_word_example(self, n: int, words: list, doc: dict) -> str:
        """Return a scrubbed example sentence for the Nth word."""
        if n == -1:
            n = len(words)
        if n < 1 or n > len(words):
            return f"No word #{n} — sentence has {len(words)} words."
        target = re.sub(r'[^a-zA-Z]', '', words[n-1])
        suf    = {1:"st",2:"nd",3:"rd"}.get(n if n < 20 else n % 10, "th")
        label  = f"{n}{suf}"
        wd = _word_info(target, doc)
        example = (wd.get("example") or "") if wd else ""
        if example:
            scrubbed = self._scrub(example[:200])
            return f"**{label} word** — example: _{scrubbed}_"
        return f"**{label} word** — no example available."

    def _resp_rag_fallback(self, msg: str, doc: dict, words: list) -> str:
        # Search ONLY within the current sentence's doc — never the whole KB
        expl = doc.get("sentence_explanation", "")
        topic = doc.get("topic", "")
        keywords = doc.get("keywords", [])

        # Try to match any keyword from the current doc to the user's message
        msg_lower = msg.lower()
        matched_kw = [kw for kw in keywords if kw.lower() in msg_lower]
        if matched_kw and expl:
            return f"Related context: {self._scrub(expl[:180])}…"

        # Check word_details for any word mentioned in the query
        for k, v in (doc.get("word_details") or {}).items():
            if k.lower() in msg_lower and v:
                defn = self._scrub((v.get("definition") or "")[:180])
                if defn:
                    syns = [s for s in v.get("synonyms", [])[:4]
                            if s.lower() != k.lower()]
                    out = f"This word ({v.get('pos','')}): {defn}"
                    if syns:
                        out += f"\n📖 Synonyms: {', '.join(syns)}"
                    return out

        # Last resort: rotating hint scoped to current sentence
        return self._resp_rotating_hint(words, doc)


# ── factory ────────────────────────────────────────────────────────────────
def create_chatbot(kb_path: str = KB_PATH) -> RAGChatbot:
    return RAGChatbot(kb_path=kb_path)


# ── test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot = create_chatbot()
    sentence = "Wisdom deepens compassion"
    print(bot.start_session(sentence))
    print()

    tests = [
        "3rd word definition",        # must NOT say 'Compassion'
        "what does compassion mean",  # must NOT say 'Compassion'
        "synonyms for wisdom",        # must NOT say 'Wisdom'
        "which words do you have definitions for?",
        "what is the topic?",
        "explain the sentence",       # must scrub 'Wisdom' and 'compassion'
        "how many words?",
        "give me a hint",
        "just tell me the answer",
    ]
    for q in tests:
        print(f"\nQ: {q}")
        print(f"A: {bot.chat(q)}")
