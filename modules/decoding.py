import torch
import torch.nn.functional as F

def constrained_log_probs(logits, sequences, bos, eos, pad, first_step,
                           repetition_penalty=1.0, no_repeat_ngram_size=0):
    if repetition_penalty < 1 or no_repeat_ngram_size < 0:
        raise ValueError('Invalid decoding penalty')
    scores = logits.float().clone()
    if not torch.isfinite(scores).all():
        raise FloatingPointError('Non-finite decoder logits')
    for row, sequence in enumerate(sequences.tolist()):
        for token in set(sequence)-{bos, eos, pad}:
            scores[row, token] = (scores[row, token]*repetition_penalty if scores[row, token]<0
                                  else scores[row, token]/repetition_penalty)
        n = no_repeat_ngram_size
        if n and len(sequence)+1 >= n:
            prefix = tuple(sequence[-(n-1):]) if n>1 else ()
            banned = {sequence[i+n-1] for i in range(len(sequence)-n+1)
                      if tuple(sequence[i:i+n-1]) == prefix}
            # EOS must remain a legal exit if all content words are blocked.
            for token in banned-{bos, eos, pad}:
                scores[row, token] = -float('inf')
    scores[:, [bos, pad]] = -float('inf')
    if first_step:
        scores[:, eos] = -float('inf')
    if not torch.isfinite(scores).any(dim=-1).all():
        raise ValueError('Decoding constraints leave no legal next token')
    return F.log_softmax(scores, dim=-1)

@torch.no_grad()
def greedy_search(next_logits, batch_size, device, bos, eos, pad, max_len,
                  repetition_penalty=1.0, no_repeat_ngram_size=0):
    sequences = torch.full((batch_size,1),bos,dtype=torch.long,device=device)
    finished = torch.zeros(batch_size,dtype=torch.bool,device=device)
    for step in range(max_len):
        logp = constrained_log_probs(next_logits(sequences),sequences,bos,eos,pad,step==0,
                                     repetition_penalty,no_repeat_ngram_size)
        token = logp.argmax(dim=-1)
        token = torch.where(finished,torch.full_like(token,pad),token)
        sequences = torch.cat([sequences,token.unsqueeze(1)],dim=1)
        finished |= token == eos
        if finished.all(): break
    return sequences

@torch.no_grad()
def beam_search(next_logits, batch_size, device, bos, eos, pad, max_len, beam_size,
                length_alpha=1.0, repetition_penalty=1.0, no_repeat_ngram_size=0):
    if beam_size < 2 or max_len < 1 or length_alpha < 0:
        raise ValueError('Invalid beam search settings')
    k = beam_size
    seqs = torch.full((batch_size,k,1),bos,dtype=torch.long,device=device)
    alive = torch.full((batch_size,k),-float('inf'),device=device)
    alive[:,0] = 0
    finished = [[] for _ in range(batch_size)]
    stopped = [False]*batch_size
    penalty = lambda n: ((5.0+n)/6.0)**length_alpha
    for step in range(max_len):
        flat = seqs.reshape(batch_size*k,-1)
        logp = constrained_log_probs(next_logits(flat),flat,bos,eos,pad,step==0,
                                     repetition_penalty,no_repeat_ngram_size)
        vocab = logp.size(-1)
        if k > vocab-2:
            raise ValueError('beam_size exceeds the number of available vocabulary tokens')
        candidates = logp.view(batch_size,k,vocab)+alive.unsqueeze(-1)
        scores, ids = candidates.view(batch_size,-1).topk(min(2*k,k*vocab),dim=-1)
        new_seqs = torch.cat([seqs,torch.full((batch_size,k,1),pad,dtype=torch.long,device=device)],dim=-1)
        new_alive = torch.full_like(alive,-float('inf'))
        for b in range(batch_size):
            if stopped[b]: continue
            retained = 0
            for raw_score, index in zip(scores[b].tolist(),ids[b].tolist()):
                if raw_score == -float('inf'): continue
                parent, token = divmod(index,vocab)
                sequence = torch.cat([seqs[b,parent],seqs.new_tensor([token])])
                if token == eos:
                    finished[b].append((raw_score/penalty(step+1),sequence))
                elif retained < k:
                    new_seqs[b,retained] = sequence
                    new_alive[b,retained] = raw_score
                    retained += 1
            finished[b] = sorted(finished[b],key=lambda x:x[0],reverse=True)[:k]
            # Future log probabilities are <=0. With alpha>=0, this is an
            # optimistic score bound even after maximum possible length reward.
            upper = new_alive[b].max().item()/penalty(max_len)
            stopped[b] = bool(finished[b] and finished[b][0][0] >= upper)
        seqs, alive = new_seqs, new_alive
        if all(stopped): break
    result=[]
    for b in range(batch_size):
        options=list(finished[b])
        if not stopped[b]:
            for j in range(k):
                if torch.isfinite(alive[b,j]):
                    options.append((alive[b,j].item()/penalty(seqs.size(-1)-1),seqs[b,j]))
        if not options:
            raise RuntimeError('Beam search produced no finite hypothesis')
        result.append(max(options,key=lambda x:x[0])[1])
    length=max(s.numel() for s in result)
    output=torch.full((batch_size,length),pad,dtype=torch.long,device=device)
    for b,sequence in enumerate(result): output[b,:len(sequence)]=sequence
    return output

def sample_encoded(self, att_feats_seq, time_lengths=None, att_masks_seq=None,
                   context_seq=None, att_feats2_seq=None, att_masks2_seq=None,
                   max_len=100, beam_size=1, repetition_penalty=1.0,
                   length_alpha=1.0, no_repeat_ngram_size=0):
    from .encoder_decoder import subsequent_mask
    hidden,mask,memory,_,_=self._encode_longitudinal_context(att_feats_seq,time_lengths,
                att_masks_seq,context_seq,att_feats2_seq,att_masks2_seq)
    B=hidden.size(0)
    if beam_size>1:
        hidden=hidden.repeat_interleave(beam_size,0)
        mask=mask.repeat_interleave(beam_size,0)
        memory=memory.repeat_interleave(beam_size,0)
    def next_logits(sequences):
        target_mask=subsequent_mask(sequences.size(1)).to(hidden.device).expand(sequences.size(0),-1,-1)
        decoded=self.model.decoder(self.model.tgt_embed(sequences),hidden,mask,target_mask,memory)
        return self.logit(decoded[:,-1])
    common=dict(next_logits=next_logits,batch_size=B,device=hidden.device,bos=self.args.bos_idx,
                eos=self.args.eos_idx,pad=self.args.pad_idx,max_len=max_len,
                repetition_penalty=repetition_penalty,no_repeat_ngram_size=no_repeat_ngram_size)
    return beam_search(**common,beam_size=beam_size,length_alpha=length_alpha) if beam_size>1 else greedy_search(**common)
