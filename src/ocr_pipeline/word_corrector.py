import math
import re
from importlib.resources import files

_CORE = re.compile(r"^([^A-Za-z]*)([A-Za-z]+(?:'[A-Za-z]+)?)([^A-Za-z]*)$")


class WordCorrector:
    """Fixes OCR misreads one word at a time and leaves every other word untouched.

    Replaces the whole-document LLM rewrite, which on the benchmark capitalised
    line starts, swapped in synonyms and invented words. Here only words that
    look misread are touched:
      - non-dictionary words ("oudget"), with SymSpell candidates ranked by
        frequency and by how often each candidate appears next to its
        neighbours (bigram counts), so "seaules infrasture" uses context;
      - real words that don't fit their neighbours ("fir our"), only on lines
        the recognizer was unsure about, and only when a one-letter-different
        word strongly fits every known neighbour ("for our"). Raw bigram counts
        aren't enough here: "fir tree" has none while "for tree" has millions,
        so the fit is judged by conditional probability instead.
    Names, acronyms, numbers and words repeated in the same document are skipped.
    Text alone can't recover badly misread words ("infrasture" is 4 edits
    from "infrastructure"); those need an image re-read, not this.
    """

    # P(neighbour | candidate) a real-word replacement must reach on every side.
    _MIN_CONDITIONAL = 0.01

    def __init__(
        self,
        max_edit_distance: int = 2,
        real_word_max_confidence: float = 0.9,
        min_candidate_frequency: int = 200_000,
    ):
        from symspellpy import SymSpell, Verbosity

        self._verbosity = Verbosity.ALL
        self._max_edit_distance = max_edit_distance
        self._sym = SymSpell(max_dictionary_edit_distance=max_edit_distance)
        data = files("symspellpy")
        self._sym.load_dictionary(str(data / "frequency_dictionary_en_82_765.txt"), term_index=0, count_index=1)
        self._sym.load_bigram_dictionary(
            str(data / "frequency_bigramdictionary_en_243_342.txt"), term_index=0, count_index=2
        )
        self._words = self._sym.words
        self._bigrams = self._sym.bigrams
        self._real_word_max_confidence = real_word_max_confidence
        # Corrections only go *to* reasonably common words (200k ~ the 40k most
        # common); rarer targets mostly turn abbreviations into other abbreviations.
        self._min_candidate_frequency = min_candidate_frequency

    def correct_lines(self, lines: list[str], confidences: list[float] | None = None) -> list[str]:
        """`confidences` are the recognizer's per-line scores; without them, real
        words are never changed (only non-dictionary words are)."""
        tokens = [line.split() for line in lines]
        counts: dict[str, int] = {}
        for line_tokens in tokens:
            for token in line_tokens:
                counts[token] = counts.get(token, 0) + 1

        # Flattened so the first word of a line sees the last word of the previous one.
        flat = [(li, ti, tok) for li, line_tokens in enumerate(tokens) for ti, tok in enumerate(line_tokens)]
        cores = [self._core(tok) for _, _, tok in flat]
        for k, (li, ti, token) in enumerate(flat):
            parts = _CORE.match(token)
            if not parts or counts[token] > 1:
                continue
            prefix, word, suffix = parts.groups()
            # Unknown neighbours (themselves misread) say nothing about this word.
            left = self._known(cores[k - 1]) if k > 0 else None
            right = self._known(cores[k + 1]) if k + 1 < len(cores) else None
            allow_real_word = confidences is not None and confidences[li] < self._real_word_max_confidence
            replacement = self._correct_word(word, left, right, line_start=ti == 0, allow_real_word=allow_real_word)
            if replacement:
                tokens[li][ti] = prefix + replacement + suffix
                cores[k] = replacement.lower()

        return [" ".join(line_tokens) for line_tokens in tokens]

    def _correct_word(
        self, word: str, left: str | None, right: str | None, line_start: bool, allow_real_word: bool
    ) -> str | None:
        if len(word) < 3 or (len(word) > 1 and word.isupper()):
            return None
        if word[0].isupper() and not line_start:
            return None

        lower = word.lower()
        if lower in self._words:
            best = self._real_word_fix(lower, left, right) if allow_real_word else None
        else:
            best = self._split_run_together(lower) or self._best_candidate(lower, left, right)
        if best is None or best == lower:
            return None
        return best.capitalize() if word[0].isupper() else best

    def _split_run_together(self, word: str) -> str | None:
        # GOT-OCR2.0 sometimes drops the spaces in a line ("callmeifthedeliveryisdelayed").
        # Split only if it divides exactly into common words, with no letters changed.
        if len(word) < 8:
            return None
        parts = self._sym.word_segmentation(word, max_edit_distance=0).corrected_string.split()
        if len(parts) < 2 or "".join(parts) != word:
            return None
        if any(self._words.get(part, 0) < self._min_candidate_frequency for part in parts):
            return None
        return " ".join(parts)

    def _best_candidate(self, word: str, left: str | None, right: str | None) -> str | None:
        # Two-letter changes on short words turn abbreviations into other words ("recvd" -> "read").
        max_distance = 1 if len(word) <= 5 else self._max_edit_distance
        suggestions = [
            s
            for s in self._sym.lookup(word, self._verbosity, max_edit_distance=max_distance)
            if s.count >= self._min_candidate_frequency
        ]
        if not suggestions:
            return None
        return max(suggestions, key=lambda s: self._score(s.term, s.distance, left, right)).term

    def _real_word_fix(self, word: str, left: str | None, right: str | None) -> str | None:
        neighbours = [n for n in (left, right) if n]
        if not neighbours or self._context(word, left, right) > 0:
            return None
        fits = [
            s
            for s in self._sym.lookup(word, self._verbosity, max_edit_distance=1)
            if s.distance == 1 and s.count >= self._min_candidate_frequency and self._fits_all(s.term, left, right)
        ]
        if not fits:
            return None
        return max(fits, key=lambda s: self._score(s.term, s.distance, left, right)).term

    def _fits_all(self, term: str, left: str | None, right: str | None) -> bool:
        if left and self._bigrams.get(f"{left} {term}", 0) / self._words[left] < self._MIN_CONDITIONAL:
            return False
        if right and self._bigrams.get(f"{term} {right}", 0) / self._words[term] < self._MIN_CONDITIONAL:
            return False
        return True

    def _known(self, word: str | None) -> str | None:
        return word if word and word in self._words else None

    def _context(self, term: str, left: str | None, right: str | None) -> int:
        total = 0
        if left:
            total += self._bigrams.get(f"{left} {term}", 0)
        if right:
            total += self._bigrams.get(f"{term} {right}", 0)
        return total

    def _score(self, term: str, distance: int, left: str | None, right: str | None) -> float:
        context = 0.0
        if left:
            context += math.log10(1 + self._bigrams.get(f"{left} {term}", 0))
        if right:
            context += math.log10(1 + self._bigrams.get(f"{term} {right}", 0))
        return math.log10(1 + self._words.get(term, 0)) + 2 * context - 1.5 * distance

    @staticmethod
    def _core(token: str) -> str | None:
        parts = _CORE.match(token)
        return parts.group(2).lower() if parts else None
