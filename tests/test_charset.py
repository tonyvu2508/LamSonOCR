import pytest


def test_charset_init():
    from config.charset import Charset
    cs = Charset()
    assert cs.num_classes > 0
    assert cs.blank_idx == 0


def test_charset_encode_decode_digits():
    from config.charset import Charset
    cs = Charset()
    text = "123"
    encoded = cs.encode(text)
    assert isinstance(encoded, list)
    assert all(isinstance(i, int) for i in encoded)
    decoded = cs.decode(encoded)
    assert decoded == text


def test_charset_encode_decode_japanese():
    from config.charset import Charset
    cs = Charset()
    text = "R03 02月"
    encoded = cs.encode(text)
    decoded = cs.decode(encoded)
    assert decoded == text


def test_charset_encode_unknown_char():
    from config.charset import Charset
    cs = Charset()
    # Unknown char should be skipped or mapped to UNK
    encoded = cs.encode("€")  # Euro sign not in charset
    assert isinstance(encoded, list)


def test_charset_decode_with_blanks():
    from config.charset import Charset
    cs = Charset()
    # CTC output often has blanks and repeats
    text = "AB"
    encoded = cs.encode(text)
    # Simulate CTC output: A, blank, blank, B
    ctc_output = [encoded[0], 0, 0, encoded[1]]
    decoded = cs.ctc_decode(ctc_output)
    assert decoded == text


def test_charset_decode_with_repeats():
    from config.charset import Charset
    cs = Charset()
    text = "AA"
    encoded = cs.encode(text)
    # CTC: A, A should collapse to "A", but A, blank, A should be "AA"
    ctc_no_sep = [encoded[0], encoded[0]]
    assert cs.ctc_decode(ctc_no_sep) == "A"
    ctc_with_sep = [encoded[0], 0, encoded[0]]
    assert cs.ctc_decode(ctc_with_sep) == "AA"
