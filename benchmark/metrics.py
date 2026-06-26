"""OCR evaluation metrics: CER, WER, Sequence Accuracy."""
import editdistance


def compute_cer(prediction: str, ground_truth: str) -> float:
    """Compute Character Error Rate.

    CER = edit_distance(pred, gt) / len(gt)
    Returns 0.0 if both empty, 1.0 if gt is empty but pred is not.
    """
    if len(ground_truth) == 0:
        return 0.0 if len(prediction) == 0 else 1.0
    distance = editdistance.eval(prediction, ground_truth)
    return min(distance / len(ground_truth), 1.0)


def compute_wer(prediction: str, ground_truth: str) -> float:
    """Compute Word Error Rate.

    WER = edit_distance(pred_words, gt_words) / len(gt_words)
    """
    pred_words = prediction.split()
    gt_words = ground_truth.split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    distance = editdistance.eval(pred_words, gt_words)
    return min(distance / len(gt_words), 1.0)


def compute_sequence_accuracy(
    predictions: list[str], ground_truths: list[str]
) -> float:
    """Compute exact-match sequence accuracy.

    Returns fraction of predictions that exactly match ground truth.
    """
    if len(predictions) == 0:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    return correct / len(predictions)
