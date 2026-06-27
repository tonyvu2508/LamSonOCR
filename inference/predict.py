"""OCR inference: predict text from images."""
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF

from model.crnn import CRNN
from config.charset import Charset


class OCRPredictor:
    """Load a trained CRNN model and predict text from images."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "cpu",
        img_height: int = 32,
        img_max_width: int = 256,
    ):
        self.device = torch.device(device)
        self.img_height = img_height
        self.img_max_width = img_max_width

        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        vocab = checkpoint.get("vocab")
        self.charset = Charset(vocab=vocab)
        
        num_classes = checkpoint.get("charset_size", self.charset.num_classes)
        self.model = CRNN(num_classes=num_classes)
        
        # Strip _orig_mod. prefix from checkpoint if present (caused by torch.compile on RunPod)
        state_dict = checkpoint["model_state_dict"]
        has_prefix = any(k.startswith("_orig_mod.") for k in state_dict.keys())
        if has_prefix:
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, image) -> torch.Tensor:
        """Preprocess a single image for the model.

        Args:
            image: PIL.Image or file path

        Returns:
            Tensor of shape (1, 1, H, W)
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        image = image.convert("L")

        # Resize to target height, maintain aspect ratio
        w, h = image.size
        new_w = max(1, int(w * self.img_height / h))
        new_w = min(new_w, self.img_max_width)
        image = image.resize((new_w, self.img_height), Image.BILINEAR)

        tensor = TF.to_tensor(image)  # (1, H, W)
        return tensor.unsqueeze(0)  # (1, 1, H, W)

    @torch.no_grad()
    def predict(self, image) -> dict:
        """Predict text from a single image.

        Args:
            image: PIL.Image or file path

        Returns:
            {"text": str, "confidence": float}
        """
        input_tensor = self._preprocess(image).to(self.device)
        log_probs = self.model(input_tensor)  # (T, 1, C)

        # Greedy decode with confidence
        probs = torch.exp(log_probs.squeeze(1))  # (T, C)
        max_probs, pred_indices = probs.max(dim=1)  # (T,)

        # CTC decode
        pred_text = self.charset.ctc_decode(pred_indices.tolist())

        # Confidence: average probability of non-blank predictions
        non_blank_mask = pred_indices != self.charset.blank_idx
        if non_blank_mask.any():
            confidence = max_probs[non_blank_mask].mean().item()
        else:
            confidence = 0.0

        return {
            "text": pred_text,
            "confidence": round(confidence, 4),
        }

    @torch.no_grad()
    def predict_batch(self, images: list) -> list[dict]:
        """Predict text from multiple images.

        Args:
            images: List of PIL.Image or file paths

        Returns:
            List of {"text": str, "confidence": float}
        """
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results
