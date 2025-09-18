from typing import Optional
import torch
from torch import nn


class CausalTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int,
        dim_feedforward: int, dropout: float):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Identity()
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)


    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.tok(x)
        h = self.pos(h)
        y = self.dec(h, torch.zeros_like(h) if memory is None else memory, tgt_mask=tgt_mask)
        y = self.ln(y)
        return self.head(y)


    @staticmethod
    def causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)