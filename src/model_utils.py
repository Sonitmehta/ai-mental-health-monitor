import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN1D(nn.Module):
    """
    Multi-Branch Deep Neural Network combining 1D Multi-Scale CNN sequence features
    with TF-IDF contextual feature representations for robust Risk & Emotion screening.
    """

    def __init__(
        self,
        vocab_size: int,
        feature_dim: int,
        embed_dim: int = 64,
        num_risk_classes: int = 3,
        num_emotion_classes: int = 4,
        kernel_sizes: List[int] = [2, 3, 4],
        num_filters: int = 32,
        dropout: float = 0.25
    ):
        super().__init__()
        # Sequence CNN Branch
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        seq_out_dim = num_filters * len(kernel_sizes)

        # Contextual Feature Branch (TF-IDF)
        self.feat_branch = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Fusion & Multi-Head Classification
        fused_dim = seq_out_dim + 64
        self.fc_shared = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.risk_head = nn.Linear(64, num_risk_classes)
        self.emotion_head = nn.Linear(64, num_emotion_classes)

    def forward(self, x_seq: torch.Tensor, x_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Sequence Conv Branch
        embedded = self.embedding(x_seq)  # (B, L, E)
        embedded = embedded.permute(0, 2, 1)  # (B, E, L)

        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(embedded))
            p = F.adaptive_max_pool1d(c, 1).squeeze(2)
            conv_outs.append(p)
        seq_rep = torch.cat(conv_outs, dim=1)

        # 2. Feature Branch
        feat_rep = self.feat_branch(x_feat)

        # 3. Multimodal Fusion
        fused = torch.cat([seq_rep, feat_rep], dim=1)
        shared = self.fc_shared(fused)

        risk_logits = self.risk_head(shared)
        emotion_logits = self.emotion_head(shared)

        return risk_logits, emotion_logits


class FeatureClassifierMLP(nn.Module):
    """
    Multi-Layer Perceptron trained on feature representations (TF-IDF)
    used to benchmark GAN-augmented classification performance.
    """

    def __init__(self, feature_dim: int, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_metadata(path: Path, metadata: Dict[str, Any]):
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_metadata(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
