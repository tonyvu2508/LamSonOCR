import pytest


def test_cer_identical():
    from benchmark.metrics import compute_cer
    assert compute_cer("hello", "hello") == 0.0


def test_cer_completely_wrong():
    from benchmark.metrics import compute_cer
    cer = compute_cer("abc", "xyz")
    assert cer == 1.0  # 3 substitutions / 3 chars


def test_cer_partial():
    from benchmark.metrics import compute_cer
    cer = compute_cer("helo", "hello")  # 1 insertion needed
    assert 0 < cer < 1


def test_cer_empty_prediction():
    from benchmark.metrics import compute_cer
    cer = compute_cer("", "hello")
    assert cer == 1.0


def test_cer_empty_both():
    from benchmark.metrics import compute_cer
    cer = compute_cer("", "")
    assert cer == 0.0


def test_wer():
    from benchmark.metrics import compute_wer
    assert compute_wer("hello world", "hello world") == 0.0
    assert compute_wer("hello", "world") == 1.0


def test_sequence_accuracy():
    from benchmark.metrics import compute_sequence_accuracy
    preds = ["hello", "world", "foo"]
    gts   = ["hello", "worlD", "foo"]
    acc = compute_sequence_accuracy(preds, gts)
    assert abs(acc - 2/3) < 1e-6


def test_sequence_accuracy_japanese():
    from benchmark.metrics import compute_sequence_accuracy
    preds = ["R03 02月", "5", "B"]
    gts   = ["R03 02月", "5", "B"]
    assert compute_sequence_accuracy(preds, gts) == 1.0
