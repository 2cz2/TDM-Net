import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.encoder_decoder import EncoderDecoder
from modules.visual_extractor import VisualExtractor


class R2GenModel(nn.Module):

    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)

        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f'\nTrainable parameters: {params}'

    def forward_iu_xray(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        if mode == 'train':
            output = self.encoder_decoder._forward(
                fc_feats, att_feats, targets, None, None, None
            )
        elif mode == 'sample':
            output, _ = self.encoder_decoder._forward(
                fc_feats, att_feats, None, None, None, None
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return output

    def forward_mimic_cxr(
        self,
        images,
        context,
        images2=None,
        targets=None,
        time_lengths=None,
        mode='train',
        update_opts=None,
        labels=None
    ):
        if images is None:
            raise ValueError("images cannot be None in forward_mimic_cxr")

        device = next(self.parameters()).device

        if not torch.is_tensor(images):
            images = torch.stack(images, dim=1).to(device=device)

        images = images.to(device)
        if not torch.is_floating_point(images):
            images = images.float()

        if images.dim() == 4:
            images = images.unsqueeze(1)

        if images.dim() != 5:
            raise RuntimeError(f"Expected images of dim 5 [B, T, C, H, W], got {images.shape}")

        B, T, C, H, W = images.shape

        if torch.isnan(images).any() or torch.isinf(images).any():
            images = torch.nan_to_num(images, nan=0.0, posinf=1e4, neginf=-1e4)

        if C == 1:
            images = images.repeat(1, 1, 3, 1, 1)
            B, T, C, H, W = images.shape

        images_flat = images.view(B * T, C, H, W).contiguous()
        att_feats_flat, _ = self.visual_extractor(images_flat)

        if att_feats_flat is None:
            raise RuntimeError("Visual extractor returned None for images.")

        _, L, D = att_feats_flat.shape
        att_feats_seq_tensor = att_feats_flat.view(B, T, L, D)

        att_feats2_seq_tensor = None
        if images2 is not None:
            if not torch.is_tensor(images2):
                images2 = torch.stack(images2, dim=1).to(device=device)

            images2 = images2.to(device)
            if not torch.is_floating_point(images2):
                images2 = images2.float()

            if images2.dim() == 4:
                images2 = images2.unsqueeze(1)

            if images2.dim() != 5:
                raise RuntimeError(f"Expected images2 of dim 5 [B, T, C, H, W], got {images2.shape}")

            B2, T2, C2, H2, W2 = images2.shape

            if torch.isnan(images2).any() or torch.isinf(images2).any():
                images2 = torch.nan_to_num(images2, nan=0.0, posinf=1e4, neginf=-1e4)

            if C2 == 1:
                images2 = images2.repeat(1, 1, 3, 1, 1)
                B2, T2, C2, H2, W2 = images2.shape

            images2_flat = images2.view(B2 * T2, C2, H2, W2).contiguous()
            att2_flat, _ = self.visual_extractor(images2_flat)

            if att2_flat is None:
                raise RuntimeError("Visual extractor returned None for images2.")

            _, L2, D2 = att2_flat.shape
            att_feats2_seq_tensor = att2_flat.view(B2, T2, L2, D2)

        if context is not None and torch.is_tensor(context):
            context = context.to(device)

        if time_lengths is not None:
            if not torch.is_tensor(time_lengths):
                time_lengths = torch.tensor(time_lengths, dtype=torch.long, device=device)
            else:
                time_lengths = time_lengths.to(device)

        if labels is not None:
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels, dtype=torch.float32, device=device)
            else:
                labels = labels.to(device).float()

        output = self.encoder_decoder.forward_mimic_cxr(
            images=att_feats_seq_tensor,
            targets=targets,
            mode=mode,
            context=context,
            images2=att_feats2_seq_tensor,
            time_lengths=time_lengths,
            update_opts=update_opts,
            labels=labels
        )

        return output