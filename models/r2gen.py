import torch
import torch.nn as nn
from modules.encoder_decoder import EncoderDecoder
from modules.visual_extractor import VisualExtractor


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args, self.tokenizer = args, tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)

    def _features(self, images, lengths):
        if images.dim()!=5:
            raise ValueError('Images must have shape [B,T,C,H,W]')
        B, T, C, H, W = images.shape
        if lengths.shape!=(B,) or (lengths<1).any() or (lengths>T).any():
            raise ValueError('Invalid temporal lengths')
        slots = torch.arange(T, device=images.device).unsqueeze(0)
        valid = slots < lengths.unsqueeze(1)
        if not self.args.history_images or self.args.history_intervention in ('none', 'remove_image'):
            valid = slots == (lengths-1).unsqueeze(1)
        packed = images[valid]
        features, _ = self.visual_extractor(packed)
        L, D = features.shape[1:]
        result = features.new_zeros((B, T, L, D))
        result[valid] = features
        return result

    def forward_mimic_cxr(self, images, context, images2=None, targets=None,
                          time_lengths=None, mode='train', update_opts=None, labels=None):
        if labels is not None:
            raise ValueError('Current disease labels are permitted in loss computation only.')
        if images.dim() == 4:
            images = images.unsqueeze(1)
        if images2 is not None and images2.dim()==4:
            images2=images2.unsqueeze(1)
        if images2 is not None and images2.shape[:2]!=images.shape[:2]:
            raise ValueError('View sequences must have equal batch and temporal dimensions')
        if mode=='sample' and targets is not None:
            raise ValueError('Generation cannot receive target reports')
        if time_lengths is None:
            time_lengths = torch.full((images.size(0),), images.size(1), dtype=torch.long, device=images.device)
        features = self._features(images, time_lengths)
        features2 = self._features(images2, time_lengths) if images2 is not None else None
        return self.encoder_decoder.forward_mimic_cxr(images=features, images2=features2,
                    targets=targets, context=context, time_lengths=time_lengths,
                    mode=mode, update_opts=update_opts, labels=None)

    forward = forward_mimic_cxr
