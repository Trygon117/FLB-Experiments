import torch
import torch.nn as nn

class BaselineTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, num_layers=4, num_heads=4, expansion=2, window_size=64, causal=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.causal = causal

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(window_size, hidden_dim)

        # Scaled parameter matching the FLB prediction slot initialization
        init_scale = 1.0 / (hidden_dim ** 0.5)
        self.prediction_slot = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * init_scale
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * expansion,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape

        token_embeds = self.token_emb(x)
        positions = torch.arange(seq_len, device=x.device)
        h = token_embeds + self.pos_emb(positions)

        if self.causal:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
            h = self.encoder(h, mask=mask, is_causal=True)
        else:
            h = self.encoder(h, mask=None, is_causal=False)

        h = self.final_norm(h)
        logits = self.head(h)

        return logits