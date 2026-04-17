import torch
import torch.nn as nn


class EmotionToTextAdapter(nn.Module):
    """
    Lightweight adapter that maps low-dim condition features to
    MusicGen text-encoder hidden states.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 768, output_seq_len: int = 32):
        super().__init__()
        self.output_seq_len = output_seq_len
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * output_seq_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, input_dim]
        y = self.net(x)
        return y.view(x.shape[0], self.output_seq_len, self.hidden_dim)

