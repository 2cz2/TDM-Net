from __future__ import absolute_import, division, print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .att_model import pack_wrapper, AttModel
from .cross_modal_attention import CrossModalTransformer


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).bool()
    return ~subsequent_mask


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(1)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )
        x = x.transpose(1, 2).contiguous().view(
            nbatches, -1, self.h * self.d_k
        )

        return self.linears[-1](x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Word_Encoder(Encoder):
    pass


class Word_EncoderLayer(EncoderLayer):
    pass


class RelationalMemory(nn.Module):
    def __init__(self, num_slots, d_model, num_heads):
        super(RelationalMemory, self).__init__()
        self.num_slots = num_slots
        self.d_model = d_model
        self.num_heads = num_heads

        self.mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.mlp = PositionwiseFeedForward(
            d_model, d_model * 4, dropout=0.1
        )
        self.mlp_norm = LayerNorm(d_model)
        self.slots = nn.Parameter(torch.randn(num_slots, d_model))
        self.dropout = nn.Dropout(0.1)

    def init_memory(self, batch_size):
        return self.slots.unsqueeze(0).repeat(batch_size, 1, 1)

    def forward(self, query, memory, query_mask=None):
        memory_sa = self.norm1(
            memory + self.dropout(self.mha(memory, memory, memory))
        )
        memory_ctx = self.norm2(
            memory_sa + self.dropout(
                self.mha(memory_sa, query, query, mask=query_mask)
            )
        )
        new_memory = self.mlp_norm(
            memory_ctx + self.dropout(self.mlp(memory_ctx))
        )
        return new_memory.view(new_memory.size(0), -1)


class ConditionalLayerNorm(nn.Module):
    def __init__(self, d_model, rm_num_slots, rm_d_model, eps=1e-6):
        super(ConditionalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

        dim_flat = rm_num_slots * rm_d_model
        self.mlp_gamma = nn.Sequential(
            nn.Linear(dim_flat, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.mlp_beta = nn.Sequential(
            nn.Linear(dim_flat, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x, memory):
        delta_gamma = self.mlp_gamma(memory).unsqueeze(1)
        delta_beta = self.mlp_beta(memory).unsqueeze(1)

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return (
                (self.gamma + delta_gamma) * (x - mean) / (std + self.eps)
                + (self.beta + delta_beta)
        )


class ConditionalSublayerConnection(nn.Module):
    def __init__(self, size, dropout, rm_num_slots, rm_d_model):
        super(ConditionalSublayerConnection, self).__init__()
        self.norm = ConditionalLayerNorm(size, rm_num_slots, rm_d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, memory):
        return x + self.dropout(sublayer(self.norm(x, memory)))


class DecoderLayer(nn.Module):
    def __init__(
            self,
            size,
            self_attn,
            src_attn,
            feed_forward,
            dropout,
            rm_num_slots,
            rm_d_model
    ):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.d_model = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(
            ConditionalSublayerConnection(
                size, dropout, rm_num_slots, rm_d_model
            ),
            3
        )

    def forward(self, x, context_fused, context_attn_mask, tgt_mask, memory):
        m = context_fused
        x = self.sublayer[0](
            x, lambda x: self.self_attn(x, x, x, tgt_mask), memory
        )
        x = self.sublayer[1](
            x, lambda x: self.src_attn(x, m, m, context_attn_mask), memory
        )
        return self.sublayer[2](x, self.feed_forward, memory)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, context, context_mask, tgt_mask, memory):
        for layer in self.layers:
            x = layer(x, context, context_mask, tgt_mask, memory)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, encoder, encoder_word, decoder, src_embed, tgt_embed, rm):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.encoder_word = encoder_word
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.rm = rm
        self.attn = MultiHeadedAttention(8, 512, dropout=0.1)

    def forward(
            self,
            src,
            tgt,
            src_mask,
            tgt_mask,
            context,
            mask,
            att_feats2,
            att_masks2
    ):
        return self.decode(
            self.encode(src, src_mask),
            self.encode(att_feats2, att_masks2),
            self.encode_word(context, mask),
            src_mask,
            att_masks2,
            tgt,
            tgt_mask,
            mask
        )

    def encode_word(self, tgt, tgt_mask):
        return self.encoder_word(self.tgt_embed(tgt), tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(
            self,
            feat_curr,
            feat_hist_img,
            feat_hist_txt,
            src_mask,
            hist_mask,
            tgt,
            tgt_mask,
            txt_mask
    ):
        if hist_mask is None:
            B = feat_hist_img.size(0)
            L2 = feat_hist_img.size(1)
            hist_mask = torch.ones(
                (B, 1, L2),
                dtype=torch.bool,
                device=feat_hist_img.device
            )

        context_fused = torch.cat([feat_curr, feat_hist_img], dim=1)

        src_mask_3d = src_mask.unsqueeze(1) if src_mask.dim() == 2 else src_mask
        hist_mask_3d = (
            hist_mask.unsqueeze(1) if hist_mask.dim() == 2 else hist_mask
        )

        context_attn_mask = torch.cat([src_mask_3d, hist_mask_3d], dim=2)

        memorym = self.rm.init_memory(feat_curr.size(0)).to(feat_curr)
        memorym = self.rm(context_fused, memorym, query_mask=context_attn_mask)

        return self.decoder(
            self.tgt_embed(tgt),
            context_fused,
            context_attn_mask,
            tgt_mask,
            memorym
        )


class EncoderDecoder(AttModel):
    def make_model(self, tgt_vocab):
        c = copy.deepcopy
        attn = MultiHeadedAttention(
            self.num_heads, self.d_model, dropout=self.dropout
        )
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        rm = RelationalMemory(
            num_slots=self.rm_num_slots,
            d_model=self.rm_d_model,
            num_heads=self.rm_num_heads
        )

        model = Transformer(
            Encoder(
                EncoderLayer(
                    self.d_model, c(attn), c(ff), self.dropout
                ),
                self.num_layers
            ),
            Word_Encoder(
                Word_EncoderLayer(
                    self.d_model, c(attn), c(ff), self.dropout
                ),
                self.num_layers
            ),
            Decoder(
                DecoderLayer(
                    self.d_model,
                    c(attn),
                    c(attn),
                    c(ff),
                    self.dropout,
                    self.rm_num_slots,
                    self.rm_d_model
                ),
                self.args.num_layers
            ),
            lambda x: x,
            nn.Sequential(
                Embeddings(self.d_model, tgt_vocab),
                c(position)
            ),
            rm
        )

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return model

    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout

        self.rm_num_slots = getattr(args, 'rm_num_slots', 3)
        self.rm_num_heads = getattr(args, 'rm_num_heads', 8)
        self.rm_d_model = getattr(args, 'rm_d_model', 512)

        tgt_vocab = self.vocab_size + 1
        self.model = self.make_model(tgt_vocab)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

        self.cross_attn = CrossModalTransformer(
            d_model=args.d_model,
            num_heads=args.num_heads
        )

        num_temporal_layers = getattr(args, 'num_temporal_layers', 1)
        self.temporal_pos = PositionalEncoding(self.d_model, self.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_temporal_layers
        )

        self.temporal_fuse = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.history_to_cond_memory = nn.Sequential(
            nn.Linear(self.d_model, self.rm_d_model * self.rm_num_slots),
            nn.ReLU(),
            nn.Linear(
                self.rm_d_model * self.rm_num_slots,
                self.rm_d_model * self.rm_num_slots
            )
        )

        dim_mem = self.rm_d_model * self.rm_num_slots
        self.memory_gate = nn.Sequential(
            nn.Linear(dim_mem * 2, dim_mem),
            nn.Sigmoid()
        )

        self.text_gate = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.Sigmoid()
        )

        self.rm_input_proj = nn.Linear(self.d_model, self.rm_d_model)

        self.max_time_steps = getattr(args, 'max_time_steps', 50)
        self.step_time_embed = nn.Embedding(self.max_time_steps, self.d_model)
        self.empty_history_embedding = nn.Parameter(
            torch.randn(1, 1, self.d_model)
        )
        nn.init.normal_(self.empty_history_embedding, mean=0, std=0.02)
        self.num_labels = 14
        self.classifier = nn.Linear(self.d_model, self.num_labels)

        self.clinical_embedding = nn.Parameter(
            torch.randn(self.num_labels, self.d_model)
        )
        nn.init.normal_(self.clinical_embedding, mean=0, std=0.02)

        self.clinical_prompt_threshold = getattr(
            args, 'clinical_prompt_threshold', 0.52
        )

    def _prepare_feature_forward(
            self,
            att_feats,
            att_masks=None,
            seq=None,
            context=None,
            att_feats2=None,
            att_masks2=None
    ):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats_p = self.att_embed(att_feats)

        att_feats2_p = None
        att_masks2_out = None
        if att_feats2 is not None:
            att_feats2, att_masks2_out = self.clip_att(att_feats2, att_masks2)
            att_feats2_p = self.att_embed(att_feats2)
        else:
            att_masks2_out = att_masks2

        seq_mask = None
        if seq is not None:
            seq_mask = (seq.data > 0).unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask.device)

        context_trim = None
        seq_mask1 = None
        text_mask_2d = None
        if context is not None:
            if context.dim() == 1:
                context = context.unsqueeze(1)
            context_trim = context[:, :self.args.max_seq_length].to(att_feats_p.device)
            text_mask_2d = (context_trim > 0).bool()
            seq_mask1 = text_mask_2d.unsqueeze(1).to(att_feats_p.device)

        return (
            att_feats_p,
            seq,
            context_trim,
            att_masks,
            seq_mask,
            seq_mask1,
            att_feats2_p,
            att_masks2_out,
            text_mask_2d,
        )

    def _crossmodal_step(
            self,
            att_feats,
            context=None,
            att_feats2=None,
            att_masks=None,
            att_masks2=None,
            targets=None
    ):
        (
            att_feats_p,
            _,
            context_trim,
            att_masks,
            _,
            seq_mask1,
            att_feats2_p,
            att_masks2,
            text_mask_2d,
        ) = self._prepare_feature_forward(
            att_feats,
            att_masks=att_masks,
            seq=None,
            context=context,
            att_feats2=att_feats2,
            att_masks2=att_masks2
        )

        if att_masks is not None:
            if att_masks.dtype != torch.bool:
                att_masks = att_masks.bool()
            if att_masks.dim() == 3:
                att_masks = att_masks.squeeze(1)
        else:
            att_masks = torch.ones(
                att_feats_p.shape[:2],
                dtype=torch.bool,
                device=att_feats_p.device
            )

        memory_img = self.model.encode(att_feats_p, att_masks)
        memory_text = None
        aligned_text = None
        att_masks2_used = None

        if context_trim is not None:
            memory_text = self.model.encode_word(context_trim, seq_mask1)
        else:
            B = memory_img.size(0)
            memory_text = self.empty_history_embedding.expand(B, -1, -1).to(memory_img.device)
            seq_mask1 = torch.ones(
                (B, 1, 1),
                dtype=torch.bool,
                device=memory_img.device
            )
            text_mask_2d = torch.ones(
                (B, 1),
                dtype=torch.bool,
                device=memory_img.device
            )

        if att_feats2_p is not None:
            if att_masks2 is not None:
                if att_masks2.dtype != torch.bool:
                    att_masks2 = att_masks2.bool()
                if att_masks2.dim() == 3:
                    att_masks2 = att_masks2.squeeze(1)
            else:
                att_masks2 = torch.ones(
                    att_feats2_p.shape[:2],
                    dtype=torch.bool,
                    device=att_feats2_p.device
                )

            memory_img2 = self.model.encode(att_feats2_p, att_masks2)
            att_masks2_used = att_masks2
        else:
            memory_img2 = memory_img
            att_masks2_used = torch.ones(
                (memory_img.size(0), memory_img.size(1)),
                dtype=torch.bool,
                device=memory_img.device
            )

        key_padding_mask = ~att_masks2_used
        aligned_text = self.cross_attn(
            memory_text,
            memory_img2,
            memory_img2,
            key_padding_mask=key_padding_mask
        )

        return memory_img, aligned_text, att_masks, att_masks2_used, text_mask_2d

    def masked_mean(self, tensor, mask, dim):
        if tensor.size(1) == 0:
            return torch.zeros(
                (tensor.size(0), tensor.size(-1)),
                device=tensor.device,
                dtype=tensor.dtype
            )
        mask_broadcast = mask.unsqueeze(-1).float()
        sum_tensor = (tensor * mask_broadcast).sum(dim=dim)
        count = mask_broadcast.sum(dim=dim).clamp(min=1e-9)
        return sum_tensor / count

    def _split_temporal_tensor(self, x, min_dim=4):
        if x is None:
            return None
        if torch.is_tensor(x) and x.dim() >= min_dim:
            return [x[:, t] for t in range(x.size(1))]
        return x

    def _build_clinical_prompt(self, cls_logits, labels=None):
        if self.training and labels is not None:
            prompt_weight = labels.float().clamp(min=0.0, max=1.0)
            prompt_mask = prompt_weight > 0
        else:
            cls_probs = torch.sigmoid(cls_logits)
            prompt_mask = cls_probs >= self.clinical_prompt_threshold
            prompt_weight = cls_probs * prompt_mask.float()

        clinical_prompt = (
                prompt_weight.unsqueeze(-1)
                * self.clinical_embedding.unsqueeze(0)
        )
        return clinical_prompt, prompt_mask.bool()

    def _pad_and_concat(self, tensor_list, mask_list, pad_mask_value=0):
        if not tensor_list:
            return None, None

        L_max = max(t.size(1) for t in tensor_list)
        padded_tensors, padded_masks = [], []

        for t, m in zip(tensor_list, mask_list):
            if t.dim() == 2:
                t = t.unsqueeze(1)

            if m is None:
                m = torch.zeros(
                    (t.size(0), t.size(1)),
                    dtype=torch.bool,
                    device=t.device
                )
            if m.dim() == 3:
                m = m.squeeze(1)

            pad_l = L_max - t.size(1)
            if pad_l > 0:
                t = F.pad(t, (0, 0, 0, pad_l))
                m = F.pad(m, (0, pad_l), value=pad_mask_value)

            padded_tensors.append(t)
            padded_masks.append(m)

        return torch.cat(padded_tensors, dim=1), torch.cat(padded_masks, dim=1)

    def _encode_longitudinal_context(
            self,
            att_feats_seq,
            time_lengths=None,
            att_masks_seq=None,
            context_seq=None,
            att_feats2_seq=None,
            att_masks2_seq=None,
            labels=None
    ):
        att_feats_seq = self._split_temporal_tensor(att_feats_seq, min_dim=4)
        att_masks_seq = self._split_temporal_tensor(att_masks_seq, min_dim=3)
        att_feats2_seq = self._split_temporal_tensor(att_feats2_seq, min_dim=4)
        att_masks2_seq = self._split_temporal_tensor(att_masks2_seq, min_dim=3)

        if context_seq is None:
            context_seq = [None] * len(att_feats_seq)
        else:
            context_seq = self._split_temporal_tensor(context_seq, min_dim=3)

        B = att_feats_seq[0].size(0)
        T = len(att_feats_seq)
        device = att_feats_seq[0].device

        per_time_z = []
        all_memory_imgs, all_att_masks = [], []
        all_aligned_texts, all_text_masks = [], []

        if time_lengths is not None:
            valid_t_mask = (
                    torch.arange(T, device=device).unsqueeze(0)
                    < time_lengths.unsqueeze(1)
            )
        else:
            valid_t_mask = torch.ones((B, T), dtype=torch.bool, device=device)

        for t in range(T):
            att_feats_t = att_feats_seq[t]
            att_masks_t = att_masks_seq[t] if (att_masks_seq and t < len(att_masks_seq)) else None
            context_t = context_seq[t] if (context_seq is not None and t < len(context_seq)) else None
            att_feats2_t = att_feats2_seq[t] if (att_feats2_seq and t < len(att_feats2_seq)) else None
            att_masks2_t = att_masks2_seq[t] if (att_masks2_seq and t < len(att_masks2_seq)) else None

            (
                memory_img_t,
                aligned_text_t,
                att_masks_t_out,
                _,
                text_mask_t
            ) = self._crossmodal_step(
                att_feats_t,
                context=context_t,
                att_feats2=att_feats2_t,
                att_masks=att_masks_t,
                att_masks2=att_masks2_t
            )

            t_safe = min(t, self.max_time_steps - 1)
            t_ids = torch.full((B,), t_safe, device=device, dtype=torch.long)
            step_embed = self.step_time_embed(t_ids).unsqueeze(1)

            memory_img_t = memory_img_t + step_embed

            valid_this_step = valid_t_mask[:, t]
            if att_masks_t_out is None:
                att_masks_t_out = torch.ones(
                    (B, memory_img_t.size(1)),
                    dtype=torch.bool,
                    device=device
                )
            elif att_masks_t_out.dim() == 3:
                att_masks_t_out = att_masks_t_out.squeeze(1)
            att_masks_t_out = att_masks_t_out & valid_this_step.unsqueeze(1)

            z_t_img = self.masked_mean(memory_img_t, att_masks_t_out, dim=1)
            z_t_txt = torch.zeros_like(z_t_img)

            if aligned_text_t is not None:
                if aligned_text_t.dim() == 2:
                    aligned_text_t = aligned_text_t.unsqueeze(1)
                aligned_text_t = aligned_text_t + step_embed
                aligned_text_t = aligned_text_t * self.text_gate(aligned_text_t)
                all_aligned_texts.append(aligned_text_t)

                if text_mask_t.dim() == 3:
                    text_mask_t = text_mask_t.squeeze(1)
                text_mask_t = text_mask_t & valid_this_step.unsqueeze(1)
                all_text_masks.append(text_mask_t)

                z_t_txt = self.masked_mean(aligned_text_t, text_mask_t, dim=1)
            else:
                dummy_txt = torch.zeros(
                    (B, 1, self.d_model),
                    device=device,
                    dtype=memory_img_t.dtype
                )
                dummy_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)
                all_aligned_texts.append(dummy_txt)
                all_text_masks.append(dummy_mask)

            z_t_fused = torch.cat([z_t_img, z_t_txt], dim=1)
            z_t_proj = self.temporal_fuse(z_t_fused)

            per_time_z.append(z_t_proj)
            all_memory_imgs.append(memory_img_t)
            all_att_masks.append(att_masks_t_out)

        Z_stack = torch.stack(per_time_z, dim=1)
        Z_input = self.temporal_pos(Z_stack)

        if time_lengths is not None:
            max_T = Z_input.size(1)
            time_pad_mask = (
                    torch.arange(max_T, device=device).unsqueeze(0)
                    >= time_lengths.unsqueeze(1)
            )
            Z_encoded = self.temporal_encoder(
                Z_input,
                src_key_padding_mask=time_pad_mask
            )
            idx = (time_lengths - 1).clamp(min=0)
        else:
            Z_encoded = self.temporal_encoder(Z_input)
            idx = torch.full((B,), T - 1, device=device, dtype=torch.long)

        batch_idx = torch.arange(B, device=device)
        temporal_summary = Z_encoded[batch_idx, idx]

        cls_logits = self.classifier(temporal_summary)
        clinical_prompt, prompt_mask = self._build_clinical_prompt(
            cls_logits, labels=labels
        )

        C_history_cond_mem = self.history_to_cond_memory(temporal_summary)
        memory_for_decoder = temporal_summary.unsqueeze(1)

        memory_all_imgs, mask_all_imgs = self._pad_and_concat(
            all_memory_imgs, all_att_masks, pad_mask_value=0
        )
        memory_all_texts, mask_all_texts = self._pad_and_concat(
            all_aligned_texts, all_text_masks, pad_mask_value=0
        )

        hidden = torch.cat(
            [
                p for p in [
                memory_all_imgs,
                memory_all_texts,
                memory_for_decoder,
                clinical_prompt
            ] if p is not None
            ],
            dim=1,
        )

        temporal_mask = torch.ones((B, 1), dtype=torch.bool, device=device)
        context_attn_mask = torch.cat(
            [
                m for m in [
                mask_all_imgs,
                mask_all_texts,
                temporal_mask,
                prompt_mask
            ] if m is not None
            ],
            dim=1,
        )
        context_attn_mask_4d = context_attn_mask.unsqueeze(1).unsqueeze(1)

        memorym = self.model.rm.init_memory(B).to(hidden)
        hidden_proj = self.rm_input_proj(hidden)
        memorym = self.model.rm(
            hidden_proj,
            memorym,
            query_mask=context_attn_mask_4d
        )

        gate = self.memory_gate(torch.cat([memorym, C_history_cond_mem], dim=-1))
        memory_final = (1 - gate) * memorym + gate * C_history_cond_mem

        return hidden, context_attn_mask_4d, memory_final, cls_logits, idx
    def forward_crossmodal_temporal(
            self,
            att_feats_seq,
            time_lengths=None,
            att_masks_seq=None,
            seq=None,
            context_seq=None,
            att_feats2_seq=None,
            att_masks2_seq=None,
            labels=None
    ):
        hidden, context_attn_mask_4d, memory_final, cls_logits, idx = self._encode_longitudinal_context(
            att_feats_seq, time_lengths, att_masks_seq, context_seq, att_feats2_seq, att_masks2_seq, labels
        )

        B = hidden.size(0)

        if seq.dim() == 2:
            target_report = seq
        else:
            target_list = []
            for i in range(B):
                curr_t = idx[i]
                curr_t = max(0, min(curr_t, seq.size(1) - 1))
                target_list.append(seq[i, curr_t])
            target_report = torch.stack(target_list)

        tgt_input = target_report[:, :-1]
        pad_id = getattr(self.args, "pad_idx", 0)

        tgt_mask = (tgt_input != pad_id).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt_input.size(-1)).to(tgt_mask.device)

        embed_out = self.model.tgt_embed(tgt_input)
        out = self.model.decoder(
            embed_out,
            hidden,
            context_attn_mask_4d,
            tgt_mask,
            memory_final
        )

        logits = self.logit(out)

        if self.model.training:
            return logits, cls_logits
        return logits

    @torch.no_grad()
    def sample_crossmodal_temporal(
            self,
            att_feats_seq,
            time_lengths=None,
            att_masks_seq=None,
            context_seq=None,
            att_feats2_seq=None,
            att_masks2_seq=None,
            max_len=60,
            greedy=True,
            temperature=1.0,
            top_k=0,
            repetition_penalty=2.0
    ):
        hidden, context_attn_mask_4d, memory_final, _, _ = self._encode_longitudinal_context(
            att_feats_seq, time_lengths, att_masks_seq, context_seq, att_feats2_seq, att_masks2_seq, labels=None
        )

        B = hidden.size(0)
        device = hidden.device
        bos = getattr(self.args, "bos_idx", 1)
        eos = getattr(self.args, "eos_idx", 2)
        pad_id = getattr(self.args, "pad_idx", 0)
        ys = torch.full((B, 1), bos, dtype=torch.long, device=device)
        finished = torch.zeros((B,), dtype=torch.bool, device=device)

        for step in range(max_len):
            tgt_mask = subsequent_mask(ys.size(1)).to(device).expand(B, -1, -1)

            embed = self.model.tgt_embed(ys)
            out = self.model.decoder(
                embed,
                hidden,
                context_attn_mask_4d,
                tgt_mask,
                memory_final
            )

            logits = self.logit(out[:, -1])
            logits[:, pad_id] = -float('inf')

            if repetition_penalty > 1.0:
                for i in range(B):
                    prev = set(ys[i].tolist())
                    prev.discard(bos)
                    prev.discard(eos)
                    prev.discard(pad_id)
                    for pid in prev:
                        if pid < logits.size(-1):
                            if logits[i, pid] < 0:
                                logits[i, pid] *= repetition_penalty
                            else:
                                logits[i, pid] /= repetition_penalty

            if greedy:
                next_token = logits.argmax(dim=-1)
                if step == 0:
                    need_fix = (next_token == eos)
                    if need_fix.any():
                        top2 = torch.topk(logits[need_fix], k=2, dim=-1).indices
                        next_token[need_fix] = top2[:, 1]
            else:
                if step == 0:
                    logits[:, eos] = -float('inf')

                probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)

                if top_k > 0:
                    topv, topi = torch.topk(probs, k=top_k, dim=-1)
                    topv = topv / topv.sum(dim=-1, keepdim=True)
                    next_token = topi.gather(
                        -1,
                        torch.multinomial(topv, 1)
                    ).squeeze(-1)
                else:
                    next_token = torch.multinomial(probs, 1).squeeze(-1)

            next_token = torch.where(
                finished,
                torch.tensor(pad_id, device=device),
                next_token
            )

            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos)

            if finished.all():
                break

        return ys

    @torch.no_grad()
    def sample_beam(
            self,
            att_feats_seq,
            time_lengths=None,
            att_masks_seq=None,
            context_seq=None,
            att_feats2_seq=None,
            att_masks2_seq=None,
            max_len=60,
            beam_size=3,
            repetition_penalty=1.2,
            length_alpha=1.0,
            no_repeat_ngram_size=3
    ):
        hidden, context_attn_mask_4d, memory_final, _, _ = self._encode_longitudinal_context(
            att_feats_seq, time_lengths, att_masks_seq, context_seq, att_feats2_seq, att_masks2_seq, labels=None
        )

        B = hidden.size(0)
        device = hidden.device
        bos = getattr(self.args, "bos_idx", 1)
        eos = getattr(self.args, "eos_idx", 2)
        pad = getattr(self.args, "pad_idx", 0)
        k = beam_size
        memory_final = memory_final.repeat_interleave(k, dim=0)
        hidden = hidden.repeat_interleave(k, dim=0)
        context_attn_mask_4d = context_attn_mask_4d.repeat_interleave(k, dim=0)

        seqs = torch.full((B * k, 1), bos, dtype=torch.long, device=device)
        beam_scores = torch.full((B, k), -1e9, device=device)
        beam_scores[:, 0] = 0.0
        done = [[] for _ in range(B)]

        def length_penalty_fn(length, alpha):
            return ((5.0 + length) / 6.0) ** alpha

        def _calc_banned_tokens(
                prev_input_ids,
                num_hypos,
                no_repeat_ngram_size,
                cur_len
        ):
            if cur_len + 1 < no_repeat_ngram_size:
                return [[] for _ in range(num_hypos)]

            generated_hyps = prev_input_ids.tolist()
            banned_tokens = [[] for _ in range(num_hypos)]

            for idx, hyp in enumerate(generated_hyps):
                ngram_idx = tuple(hyp[-(no_repeat_ngram_size - 1):])
                for i in range(len(hyp) - no_repeat_ngram_size + 1):
                    if tuple(hyp[i: i + (no_repeat_ngram_size - 1)]) == ngram_idx:
                        banned_tokens[idx].append(hyp[i + no_repeat_ngram_size - 1])
            return banned_tokens

        for step in range(max_len):
            L = seqs.size(1)
            tgt_mask = subsequent_mask(L).to(device).expand(B * k, -1, -1)

            out = self.model.decoder(
                self.model.tgt_embed(seqs),
                hidden,
                context_attn_mask_4d,
                tgt_mask,
                memory_final
            )

            logits = self.logit(out[:, -1, :])
            logits[:, pad] = -float('inf')
            if step == 0:
                logits[:, eos] = -float('inf')

            logp = F.log_softmax(logits, dim=-1)
            V = logp.size(-1)
            if repetition_penalty > 1.0:
                for i in range(B * k):
                    prev = set(seqs[i].tolist())
                    prev.discard(bos)
                    prev.discard(eos)
                    prev.discard(pad)
                    for tok in prev:
                        if logp[i, tok] < 0:
                            logp[i, tok] *= repetition_penalty
                        else:
                            logp[i, tok] /= repetition_penalty
            if no_repeat_ngram_size > 1:
                banned_tokens = _calc_banned_tokens(
                    seqs, B * k, no_repeat_ngram_size, L
                )
                for i, banned in enumerate(banned_tokens):
                    if len(banned) > 0:
                        logp[i, banned] = -float('inf')

            logp = logp.view(B, k, V)
            cand_scores = logp + beam_scores.unsqueeze(-1)
            cand_scores = cand_scores.view(B, -1)

            top_scores, top_ids = cand_scores.topk(k, dim=-1)
            next_beam = torch.div(top_ids, V, rounding_mode='floor')
            next_tok = top_ids % V

            seqs_ = seqs.view(B, k, -1)
            new_seqs = []
            new_beam_scores = top_scores.clone()

            for b in range(B):
                b_seqs = []
                for j in range(k):
                    pb = next_beam[b, j]
                    tok = next_tok[b, j]
                    ns = torch.cat([seqs_[b, pb], tok.view(1)], dim=0)
                    b_seqs.append(ns)

                    if tok.item() == eos:
                        lp = length_penalty_fn(ns.size(0), length_alpha)
                        done[b].append((new_beam_scores[b, j].item() / lp, ns))
                        new_beam_scores[b, j] = -1e9

                new_seqs.append(torch.stack(b_seqs, dim=0))

            seqs = torch.stack(new_seqs, dim=0).view(B * k, -1)
            beam_scores = new_beam_scores

            if all(len(done[b]) >= k for b in range(B)):
                break

        final = []
        for b in range(B):
            if len(done[b]) > 0:
                done[b].sort(key=lambda x: x[0], reverse=True)
                best = done[b][0][1]
            else:
                best = seqs.view(B, k, -1)[b, 0]
            final.append(best)

        maxL = max(x.size(0) for x in final)
        out = []
        for x in final:
            if x.size(0) < maxL:
                x = F.pad(x, (0, maxL - x.size(0)), value=pad)
            out.append(x)

        return torch.stack(out, dim=0)

    def forward_mimic_cxr(
            self,
            images,
            targets=None,
            mode='train',
            context=None,
            images2=None,
            time_lengths=None,
            update_opts=None,
            labels=None
    ):
        att_feats_seq = images
        if att_feats_seq.dim() == 3:
            att_feats_seq = att_feats_seq.unsqueeze(1)

        att_feats2_seq = images2
        if att_feats2_seq is not None and att_feats2_seq.dim() == 3:
            att_feats2_seq = att_feats2_seq.unsqueeze(1)

        context_seq = context

        if mode == 'train':
            if targets is None:
                raise ValueError("Train mode requires targets.")
            return self.forward_crossmodal_temporal(
                att_feats_seq=att_feats_seq,
                seq=targets,
                context_seq=context_seq,
                att_feats2_seq=att_feats2_seq,
                time_lengths=time_lengths,
                att_masks_seq=None,
                labels=labels,
            )

        elif mode == 'sample':
            opts = update_opts if update_opts is not None else {}
            beam_size = opts.get('beam_size', getattr(self.args, "beam_size", 1))
            length_alpha = opts.get(
                'length_penalty',
                getattr(self.args, "length_alpha", 1.0)
            )
            repetition_penalty = opts.get(
                'repetition_penalty',
                getattr(self.args, "repetition_penalty", 1.2)
            )
            no_repeat_ngram_size = opts.get(
                'no_repeat_ngram_size',
                getattr(self.args, "no_repeat_ngram_size", 3)
            )

            if beam_size > 1:
                return self.sample_beam(
                    att_feats_seq=att_feats_seq,
                    context_seq=context_seq,
                    att_feats2_seq=att_feats2_seq,
                    time_lengths=time_lengths,
                    max_len=60,
                    beam_size=beam_size,
                    repetition_penalty=repetition_penalty,
                    length_alpha=length_alpha,
                    no_repeat_ngram_size=no_repeat_ngram_size
                )
            else:
                return self.sample_crossmodal_temporal(
                    att_feats_seq=att_feats_seq,
                    context_seq=context_seq,
                    att_feats2_seq=att_feats2_seq,
                    time_lengths=time_lengths,
                    max_len=60,
                    greedy=True,
                    repetition_penalty=repetition_penalty
                )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _forward(self, fc_feats, att_feats, seq, context, fc_feats2, att_feats2, att_masks=None):
        return self.forward_mimic_cxr(
            images=att_feats,
            targets=seq,
            context=context,
            images2=att_feats2,
            mode='train'
        )

    def core(self, *args, **kwargs):
        return None