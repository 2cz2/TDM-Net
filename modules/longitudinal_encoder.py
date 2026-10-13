import torch
from .contract import validate_intervention


def encode_longitudinal_context(self, att_feats_seq, time_lengths=None, att_masks_seq=None,
                                context_seq=None, att_feats2_seq=None, att_masks2_seq=None,
                                labels=None):
    if labels is not None:
        raise ValueError('Target disease labels are loss targets only, never encoder inputs.')
    condition = validate_intervention(self.args, training=self.training)
    images = self._split_temporal_tensor(att_feats_seq, min_dim=4)
    images2 = self._split_temporal_tensor(att_feats2_seq, min_dim=4)
    texts = self._split_temporal_tensor(context_seq, min_dim=3)
    B, T, device = images[0].size(0), len(images), images[0].device
    lengths = time_lengths if time_lengths is not None else torch.full((B,), T, device=device, dtype=torch.long)
    if (lengths < 1).any() or (lengths > T).any():
        raise ValueError('Invalid temporal lengths')
    idx = lengths - 1
    batch = torch.arange(B, device=device)
    mode = getattr(self.args, 'history_intervention', 'factual')
    use_images = self.args.history_images and mode not in ('none', 'remove_image')
    use_text = self.args.history_reports and mode not in ('none', 'remove_report')
    valid = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)

    image_memory = []
    for t in range(T):
        im = self.att_embed(images[t])
        if images2 is not None:
            im = torch.cat([im, self.att_embed(images2[t])], dim=1)
        mask = torch.ones(im.shape[:2], dtype=torch.bool, device=device)
        image_memory.append(self.model.encode(im, mask))
    current = torch.stack(image_memory, dim=1)[batch, idx]
    current_mask = torch.ones(current.shape[:2], dtype=torch.bool, device=device)
    current_summary = current.mean(dim=1)
    cls_logits = self.classifier(current_summary) if self.args.use_dgl else None

    history_tokens, history_masks, temporal_vectors, temporal_masks = [], [], [], []
    for t in range(T):
        im = image_memory[t]
        is_previous = valid[:, t] & (idx > t) & use_images
        image_mask = is_previous.unsqueeze(1).expand(-1, im.size(1))
        position = self.step_time_embed(torch.full((B,), min(t, self.max_time_steps-1), device=device, dtype=torch.long)).unsqueeze(1)
        history_tokens.append(im + position)
        history_masks.append(image_mask)
        img_summary = self.masked_mean(im + position, image_mask, dim=1)
        text_summary = torch.zeros_like(img_summary)
        has_report = torch.zeros(B, dtype=torch.bool, device=device)
        if use_text and texts is not None and t < len(texts) and texts[t] is not None:
            ids = texts[t]
            if ids.dim() == 1:
                ids = ids.unsqueeze(1)
            report_mask = (ids != self.args.pad_idx) & valid[:, t].unsqueeze(1)
            # Context slot 0 cannot contain a prior report in this window protocol.
            if t == 0:
                report_mask = torch.zeros_like(report_mask)
            has_report = report_mask.any(dim=1)
            safe_ids = ids.clone()
            safe_mask = report_mask.clone()
            safe_ids[~has_report, 0] = self.args.bos_idx
            safe_mask[~has_report, 0] = True
            tx = self.model.encode_word(safe_ids, safe_mask.unsqueeze(1))
            if self.args.use_cmi:
                # Text-only-history ablation must not access historical images.
                keys = im if use_images else current
                tx = self.cross_attn(tx, keys, keys)
            # Direct concat has no learnable gating of text or images.
            tx = tx + position
            history_tokens.append(tx)
            history_masks.append(report_mask)
            text_summary = self.masked_mean(tx, report_mask, dim=1)
        temporal_vectors.append(self.temporal_fuse(torch.cat([img_summary, text_summary], dim=-1)))
        temporal_masks.append(is_previous | has_report)

    h_tokens = torch.cat(history_tokens, dim=1)
    h_mask = torch.cat(history_masks, dim=1)
    has_history = h_mask.any(dim=1)
    if self.args.use_te:
        z = torch.cat([current_summary.unsqueeze(1), torch.stack(temporal_vectors, dim=1)], dim=1)
        z_mask = torch.cat([torch.ones((B, 1), dtype=torch.bool, device=device),
                            torch.stack(temporal_masks, dim=1)], dim=1)
        z = self.temporal_encoder(self.temporal_pos(z), src_key_padding_mask=~z_mask)
        h_summary = z[:, 0]
        h_tokens = torch.cat([h_tokens, h_summary.unsqueeze(1)], dim=1)
        h_mask = torch.cat([h_mask, has_history.unsqueeze(1)], dim=1)
    else:
        h_summary = self.masked_mean(h_tokens, h_mask, dim=1)

    learned_gate = self.history_gate(torch.cat([current_summary, h_summary], dim=-1)) if self.args.use_mg else torch.ones((B, 1), device=device)
    gate = learned_gate
    override = getattr(self.args, 'history_gate_override', None)
    if override is not None and (self.training or not 0 <= float(override) <= 1):
        raise ValueError('Internal gate override is an evaluation-only value in [0,1]')
    if condition['mechanism'] in ('gate_closed', 'gate_open'):
        override = 0.0 if condition['mechanism'] == 'gate_closed' else 1.0
    if override is not None:
        gate = torch.full_like(gate, float(override))
    gate = gate * has_history.unsqueeze(1)
    self.last_learned_history_gate = (learned_gate * has_history.unsqueeze(1)).detach()
    self.last_history_gate = gate.detach()
    h_tokens = h_tokens * gate.unsqueeze(1)
    h_mask = h_mask & (gate > 0)
    hidden = torch.cat([current, h_tokens], dim=1)
    hidden_mask = torch.cat([current_mask, h_mask], dim=1)
    if self.args.use_dgl and condition['mechanism'] != 'prompt_off':
        prompt, prompt_mask = self._build_clinical_prompt(cls_logits)
        hidden = torch.cat([hidden, prompt], dim=1)
        hidden_mask = torch.cat([hidden_mask, prompt_mask], dim=1)

    base = self.model.rm(self.rm_input_proj(current), self.model.rm.init_memory(B).to(current),
                         query_mask=current_mask.unsqueeze(1).unsqueeze(1))
    memory_final = base
    if self.args.use_mg:
        if not (self.args.history_images or self.args.history_reports):
            # Capacity control requested by reviewers: MG applied to current evidence only.
            cond = self.history_to_cond_memory(current_summary)
            memory_gate = self.memory_gate(torch.cat([base, cond], dim=-1))
        else:
            cond = self.history_to_cond_memory(h_summary)
            memory_gate = self.memory_gate(torch.cat([base, cond], dim=-1)) * gate
        memory_final = (1-memory_gate)*base + memory_gate*cond
        self.last_memory_gate_mean = memory_gate.mean(dim=-1).detach()
    else:
        self.last_memory_gate_mean = torch.zeros(B, device=device)
    return hidden, hidden_mask.unsqueeze(1).unsqueeze(1), memory_final, cls_logits, idx
