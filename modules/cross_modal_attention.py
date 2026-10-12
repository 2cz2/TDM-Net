import torch
import torch.nn as nn


class CrossModalTransformer(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads,
            dropout=dropout, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_feats, key_feats, value_feats, key_padding_mask=None):
        assert query_feats.dim() == 3 and key_feats.dim() == 3 and value_feats.dim() == 3
        
        attn_out, _ = self.cross_attn(
            query_feats, 
            key_feats, 
            value_feats,
            key_padding_mask=key_padding_mask
        )
        
        out = self.norm(query_feats + self.dropout(attn_out))
        return out