"""CRNN (Convolutional Recurrent Neural Network) for text line recognition."""
import torch
import torch.nn as nn


class _CNNBackbone(nn.Module):
    """Lightweight CNN feature extractor.

    Input:  (B, C, 32, W)
    Output: (B, 512, 1, W') where W' = W // 4
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: (B, C, 32, W) -> (B, 64, 16, W/2)
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: (B, 64, 16, W/2) -> (B, 128, 8, W/4)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: (B, 128, 8, W/4) -> (B, 256, 8, W/4)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 4: (B, 256, 8, W/4) -> (B, 256, 4, W/4)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # Only reduce height

            # Block 5: (B, 256, 4, W/4) -> (B, 512, 2, W/4)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Block 6: (B, 512, 2, W/4) -> (B, 512, 1, W/4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),  # Collapse height to 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class CRNN(nn.Module):
    """CRNN model for text line recognition with CTC.

    Architecture: CNN backbone -> BiLSTM -> FC -> LogSoftmax

    Args:
        num_classes: Number of character classes (including CTC blank)
        img_height: Input image height (default: 32)
        img_channels: Number of input channels (1=grayscale, 3=RGB)
        rnn_hidden: Hidden size of LSTM layers
        rnn_layers: Number of LSTM layers
    """

    def __init__(
        self,
        num_classes: int,
        img_height: int = 32,
        img_channels: int = 1,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
    ):
        super().__init__()
        assert img_height == 32, "CRNN requires img_height=32"

        self.cnn = _CNNBackbone(in_channels=img_channels)
        # After CNN: (B, 512, 1, W')
        # BiLSTM input: (B, W', 512)
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1 if rnn_layers > 1 else 0,
        )
        # BiLSTM output: (B, W', rnn_hidden*2)
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            Log probabilities (T, B, num_classes) where T = W // 4
        """
        # CNN features
        features = self.cnn(x)  # (B, 512, 1, W')

        # Squeeze height, permute to (B, W', 512)
        features = features.squeeze(2)  # (B, 512, W')
        features = features.permute(0, 2, 1)  # (B, W', 512)

        # BiLSTM
        rnn_out, _ = self.rnn(features)  # (B, W', hidden*2)

        # FC + LogSoftmax
        output = self.fc(rnn_out)  # (B, W', num_classes)
        output = self.log_softmax(output)

        # Permute to (T, B, num_classes) for CTC loss
        output = output.permute(1, 0, 2)  # (T, B, num_classes)

        return output
