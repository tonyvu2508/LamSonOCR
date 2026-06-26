"""Evaluate OCR model on a dataset and produce benchmark report."""
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
from datetime import datetime

from model.crnn import CRNN
from data.dataset import OCRDataset, ocr_collate_fn
from benchmark.metrics import compute_cer, compute_wer, compute_sequence_accuracy


class Evaluator:
    """Run evaluation on a dataset and compute metrics."""

    def __init__(self, model: CRNN, charset, device: str = "cpu"):
        self.model = model.to(device)
        self.charset = charset
        self.device = torch.device(device)

    @torch.no_grad()
    def evaluate(
        self, dataset: OCRDataset, batch_size: int = 32
    ) -> dict:
        """Evaluate model on dataset.

        Returns:
            {
                "cer": float,
                "wer": float,
                "sequence_accuracy": float,
                "num_samples": int,
                "per_sample": [{"prediction": str, "ground_truth": str, "cer": float}, ...]
            }
        """
        self.model.eval()
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=ocr_collate_fn,
        )

        all_predictions = []
        all_ground_truths = []
        per_sample = []

        for images, labels, label_lengths, texts in loader:
            images = images.to(self.device)
            log_probs = self.model(images)  # (T, B, C)

            # Greedy decode
            _, preds_indices = log_probs.max(2)  # (T, B)
            preds_indices = preds_indices.permute(1, 0)  # (B, T)

            for i, (pred_seq, gt_text) in enumerate(zip(preds_indices, texts)):
                pred_text = self.charset.ctc_decode(pred_seq.tolist())
                all_predictions.append(pred_text)
                all_ground_truths.append(gt_text)

                sample_cer = compute_cer(pred_text, gt_text)
                per_sample.append({
                    "prediction": pred_text,
                    "ground_truth": gt_text,
                    "cer": sample_cer,
                })

        avg_cer = sum(compute_cer(p, g) for p, g in zip(all_predictions, all_ground_truths)) / max(len(all_predictions), 1)
        avg_wer = sum(compute_wer(p, g) for p, g in zip(all_predictions, all_ground_truths)) / max(len(all_predictions), 1)
        seq_acc = compute_sequence_accuracy(all_predictions, all_ground_truths)

        return {
            "cer": avg_cer,
            "wer": avg_wer,
            "sequence_accuracy": seq_acc,
            "num_samples": len(all_predictions),
            "per_sample": per_sample,
            "timestamp": datetime.now().isoformat(),
        }

    def save_report(self, results: dict, output_path: Path):
        """Save evaluation report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Report saved: {output_path}")
