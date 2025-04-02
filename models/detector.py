import torch
import torch.nn as nn
import torch.nn.functional as F
from .position_encoding import PositionalEncoding

class TransformerViolenceDetector(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, num_layers=4, nhead=8, dropout=0.1):
        super(TransformerViolenceDetector, self).__init__()

        # Position encoding
        self.pos_encoder = PositionalEncoding(input_dim, dropout)

        # Transformer encoder layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu'
        )

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )

        # Sequence pooling - aggregate sequence representation into a single vector
        self.attention_pool = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Softmax(dim=1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # Input shape: [batch_size, seq_len, feature_dim]
        # Convert to Transformer expected shape: [seq_len, batch_size, feature_dim]
        x = x.permute(1, 0, 2)

        # Add position encoding
        x = self.pos_encoder(x)

        # Pass through Transformer encoder
        x = self.transformer_encoder(x)

        # Convert back to [batch_size, seq_len, feature_dim]
        x = x.permute(1, 0, 2)

        # Attention pooling
        attn_weights = self.attention_pool(x)  # [batch_size, seq_len, 1]
        weighted_x = torch.bmm(attn_weights.permute(0, 2, 1), x)  # [batch_size, 1, feature_dim]
        pooled = weighted_x.squeeze(1)  # [batch_size, feature_dim]

        # Classification
        output = self.classifier(pooled)

        return output 