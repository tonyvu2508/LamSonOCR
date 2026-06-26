"""Training and model configuration."""
from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class Settings:
    # Paths
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    data_dir: Path = field(default=None)
    model_dir: Path = field(default=None)

    # Image
    img_height: int = 32
    img_max_width: int = 256
    img_channels: int = 1  # Grayscale

    # Model
    cnn_output_channels: int = 512
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 2
    rnn_dropout: float = 0.1

    # Training
    batch_size: int = 64
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    early_stop_patience: int = 10
    num_workers: int = 0  # 0 for MPS compatibility

    # Device
    device: str = field(default=None)

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.model_dir is None:
            self.model_dir = self.project_root / "checkpoints"
        if self.device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                
        if self.device == "cuda" and self.num_workers == 0:
            import multiprocessing
            self.num_workers = min(4, multiprocessing.cpu_count())
