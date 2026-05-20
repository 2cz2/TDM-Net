import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets import MimiccxrSequenceDataset


class R2DataLoader(DataLoader):

    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)
                ),
            ])

        self.dataset = MimiccxrSequenceDataset(
            args, tokenizer, self.split, transform=self.transform
        )

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
        }

        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        (images_lists,
         images2_lists,
         context_lists,
         targets_list,
         masks_list,
         labels_lists,
         label_masks_lists,
         seq_lens) = zip(*batch)

        max_T = max(len(imgs) for imgs in images_lists)
        C, H, W = images_lists[0][0].shape

        images_seq_tensor = torch.zeros(
            (batch_size, max_T, C, H, W),
            dtype=images_lists[0][0].dtype
        )
        images2_seq_tensor = torch.zeros(
            (batch_size, max_T, C, H, W),
            dtype=images_lists[0][0].dtype
        )

        for i, (imgs, imgs2) in enumerate(zip(images_lists, images2_lists)):
            for t in range(len(imgs)):
                images_seq_tensor[i, t] = imgs[t]
                if imgs2[t] is not None:
                    images2_seq_tensor[i, t] = imgs2[t]
                else:
                    images2_seq_tensor[i, t] = imgs[t]

        max_L = 0
        for seq in targets_list:
            for t_seq in seq:
                if isinstance(t_seq, torch.Tensor):
                    max_L = max(max_L, t_seq.size(0))
                else:
                    max_L = max(max_L, len(t_seq))

        targets_seq = torch.zeros((batch_size, max_T, max_L), dtype=torch.long)
        targets_masks_seq = torch.zeros((batch_size, max_T, max_L), dtype=torch.long)

        for i, (toks_seq, masks_seq) in enumerate(zip(targets_list, masks_list)):
            for t_idx in range(len(toks_seq)):
                toks = (
                    toks_seq[t_idx]
                    if isinstance(toks_seq[t_idx], torch.Tensor)
                    else torch.tensor(toks_seq[t_idx], dtype=torch.long)
                )
                mask = (
                    masks_seq[t_idx]
                    if isinstance(masks_seq[t_idx], torch.Tensor)
                    else torch.tensor(masks_seq[t_idx], dtype=torch.long)
                )
                L = toks.size(0)
                targets_seq[i, t_idx, :L] = toks
                targets_masks_seq[i, t_idx, :L] = mask

        max_ctx_L = 0
        for ctx_seq in context_lists:
            if ctx_seq is None:
                continue
            for ctx in ctx_seq:
                if ctx is None:
                    continue
                Lc = ctx.size(0) if isinstance(ctx, torch.Tensor) else len(ctx)
                max_ctx_L = max(max_ctx_L, Lc)

        if max_ctx_L == 0:
            max_ctx_L = 1

        context_seq_tensor = torch.zeros(
            (batch_size, max_T, max_ctx_L),
            dtype=torch.long
        )

        for i, ctx_seq in enumerate(context_lists):
            if ctx_seq is None:
                ctx_seq = [None] * max_T
            elif len(ctx_seq) < max_T:
                ctx_seq = ctx_seq + [None] * (max_T - len(ctx_seq))

            for t_idx in range(max_T):
                ctx = ctx_seq[t_idx]
                if ctx is None:
                    continue
                ctx_t = (
                    ctx if isinstance(ctx, torch.Tensor)
                    else torch.tensor(ctx, dtype=torch.long)
                )
                Lc = ctx_t.size(0)
                context_seq_tensor[i, t_idx, :Lc] = ctx_t

        num_classes = 14
        labels_seq_tensor = torch.zeros(
            (batch_size, max_T, num_classes),
            dtype=torch.float32
        )
        label_masks_seq_tensor = torch.zeros(
            (batch_size, max_T, num_classes),
            dtype=torch.float32
        )

        for i, (labs, lab_masks) in enumerate(zip(labels_lists, label_masks_lists)):
            for t, (lab, lab_mask) in enumerate(zip(labs, lab_masks)):
                if not isinstance(lab, torch.Tensor):
                    lab = torch.tensor(lab, dtype=torch.float32)
                if not isinstance(lab_mask, torch.Tensor):
                    lab_mask = torch.tensor(lab_mask, dtype=torch.float32)
                labels_seq_tensor[i, t] = lab
                label_masks_seq_tensor[i, t] = lab_mask

        seq_len_tensor = torch.tensor(seq_lens, dtype=torch.long)

        return {
            'images_seq': images_seq_tensor,
            'images2_seq': images2_seq_tensor,
            'context_seq': context_seq_tensor,
            'targets': targets_seq,
            'targets_masks': targets_masks_seq,
            'labels': labels_seq_tensor,
            'label_masks': label_masks_seq_tensor,
            'seq_len': seq_len_tensor,
        }