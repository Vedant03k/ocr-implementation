from ocr_pipeline.recognizer_specialist import collapse_repeats, usable


def test_rejects_refusal_on_blank_crop():
    assert usable("The image is too blurry to recognize any text content.") == ""


def test_rejects_formula_style_misreads():
    assert usable("∆V-11-38 un complete") == ""
    assert usable("\\( \\leq \\) to p") == ""


def test_keeps_ordinary_handwriting_reads():
    for text in ["KN - 3-05 uncompleted", "Ave (a+b+c)/3", "I loved the image of it", "50% & more"]:
        assert usable(text) == text


def test_collapses_decoding_loop():
    assert collapse_repeats("Ave " * 60 + "Ave") == "Ave"


def test_keeps_text_around_the_loop():
    assert collapse_repeats("Display Ave Ave Ave Ave now") == "Display Ave now"


def test_keeps_a_doubled_word():
    assert collapse_repeats("that that is") == "that that is"


def test_does_not_merge_word_prefixes():
    assert collapse_repeats("a a ab") == "a a ab"
    assert collapse_repeats("no no no-one") == "no no no-one"


def test_leaves_normal_lines_alone():
    line = "write an algorithm and draw the flowchart"
    assert collapse_repeats(line) == line
