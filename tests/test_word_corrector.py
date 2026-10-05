import pytest

from ocr_pipeline.word_corrector import WordCorrector


@pytest.fixture(scope="module")
def corrector():
    return WordCorrector()


def correct(corrector, *lines, confidence=0.5):
    return corrector.correct_lines(list(lines), [confidence] * len(lines))


@pytest.mark.parametrize(
    "misread, expected",
    [
        ("olease rush this one, job site closes Friday", "please rush this one, job site closes Friday"),
        ("blease rush this one", "please rush this one"),
        ("oudget approved for the new warehouse", "budget approved for the new warehouse"),
    ],
)
def test_fixes_non_dictionary_misreads(corrector, misread, expected):
    assert correct(corrector, misread) == [expected]


def test_fixes_real_word_from_next_lines_context(corrector):
    # "fir" -> "for" from "our" on the next line; the misread neighbour "infrasture"
    # is ignored rather than counted against the fix.
    assert correct(corrector, "seaules infrasture fir", "our Industry to find out")[0].endswith(" for")


def test_real_words_untouched_on_confident_lines(corrector):
    assert correct(corrector, "infrasture fir", "our team", confidence=0.99)[0].endswith(" fir")


def test_real_words_untouched_without_confidences(corrector):
    assert corrector.correct_lines(["infrasture fir", "our team"])[0].endswith(" fir")


@pytest.mark.parametrize(
    "line",
    [
        "the fir tree in the garden",  # rare but valid pair, absent from the bigram list
        "review with Marcus on Tuesday at 9am",  # names, numbers
        "restock by Friday, low on M8 bolts",
        "Project name: Shadow Sentinel",
        "recvd by J.T. - ok to ship",  # informal abbreviation, not an OCR error
        "Subtotal: $842.00 including tax",
    ],
)
def test_leaves_correct_text_alone(corrector, line):
    assert correct(corrector, line) == [line]


def test_splits_run_together_words(corrector):
    assert correct(corrector, "callmeifthedeliveryisdelayed") == ["call me if the delivery is delayed"]


def test_keeps_capitalisation(corrector):
    assert correct(corrector, "Olease rush this one") == ["Please rush this one"]


def test_skips_words_repeated_in_document(corrector):
    # A non-dictionary word that recurs is more likely a real term than a misread.
    assert correct(corrector, "the zorblax unit", "zorblax is ready") == ["the zorblax unit", "zorblax is ready"]
