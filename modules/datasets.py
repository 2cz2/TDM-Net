import json
import random
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from .protocol import CLASS_NAMES, audit_annotation, read_metadata, build_windows, training_labels
from .contract import intervention


class MimiccxrSequenceDataset(Dataset):
    """One target study per sample; current report is never historical context."""
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = Path(args.image_dir)
        self.tokenizer, self.transform = tokenizer, transform
        self.max_seq_length = args.max_seq_length
        if self.max_seq_length < 3:
            raise ValueError('max_seq_length must allow BOS, content and EOS')
        self.args = args
        self.split = split
        ann = json.loads(Path(args.ann_path).read_text(encoding='utf-8'))
        audit_annotation(ann)
        self.metadata = read_metadata(args.metadata_path)
        self.examples, self.statistics = build_windows(ann[split], self.metadata, args.max_history)
        self._evaluation_records = []
        token_cache = {}
        for window in self.examples:
            for rec in window:
                if rec['study_id'] not in token_cache:
                    ids = tokenizer(rec['report'])[:self.max_seq_length]
                    if ids[-1] != tokenizer.eos_id:
                        ids[-1] = tokenizer.eos_id
                    token_cache[rec['study_id']] = ids
                ids = token_cache[rec['study_id']]
                rec['ids'] = ids
            target, prior = window[-1], window[-2] if len(window)>1 else None
            self._evaluation_records.append(self._record(window))
            # Validate current supervision before spending GPU time on the first epoch.
            self._labels(target)

    def __len__(self):
        return len(self.examples)

    def _labels(self, rec):
        # Validation and test labels are never needed by the generation pipeline.
        if self.split != 'train' or not getattr(self.args, 'use_dgl', False):
            return torch.zeros(14), torch.zeros(14)
        labels, masks = training_labels(rec, self.metadata[rec['study_id']], require=True)
        return torch.tensor(labels), torch.tensor(masks)

    def _record(self, window):
        target, prior = window[-1], window[-2] if len(window)>1 else None
        return {'subject_id': target['subject_id'], 'study_id': target['study_id'],
                'previous_study_id': prior['study_id'] if prior else None,
                'history_study_ids': [r['study_id'] for r in window[:-1]],
                'reference': target['report'], 'previous_report': prior['report'] if prior else None,
                'study_time': self.metadata[target['study_id']]['timestamp'].isoformat(),
                'current_image_paths': target['image_path'][:2]}

    def evaluation_records(self):
        return [dict(r) for r in self._evaluation_records]

    def _image(self, relative):
        with Image.open(self.image_dir / relative) as f:
            image = f.convert('RGB')
        return self.transform(image) if self.transform else image

    def __getitem__(self, idx):
        window = self.examples[idx]
        target, prior = window[-1], window[-2] if len(window) > 1 else None
        mode = getattr(self.args, 'history_intervention', 'factual')
        if self.split == 'train' and random.random() < self.args.history_dropout:
            mode = 'none'
        no_history = mode == 'none' or not (self.args.history_images or self.args.history_reports)
        active_window = [target] if no_history else window
        # Load the current image before optional history, keeping its augmentation independent
        # of the number of images in this sample. No removed historical image is read.
        target_paths = target['image_path']
        current_image = self._image(target_paths[0])
        current_second = self._image(target_paths[1]) if len(target_paths)>1 and target_paths[1]!=target_paths[0] else current_image
        images, images2, context, targets, masks, labels, label_masks = [], [], [], [], [], [], []
        for t, rec in enumerate(active_window):
            paths = rec['image_path'] if isinstance(rec['image_path'], list) else [rec['image_path']]
            if rec is target:
                im, second = current_image, current_second
            elif self.args.history_images and mode != 'remove_image':
                im = self._image(paths[0])
                second = self._image(paths[1]) if len(paths)>1 and paths[1]!=paths[0] else im
            else:
                im, second = torch.zeros_like(current_image), torch.zeros_like(current_second)
            images.append(im)
            images2.append(second)
            targets.append(torch.tensor(rec['ids'], dtype=torch.long))
            masks.append(torch.ones(len(rec['ids']), dtype=torch.long))
            lab, mask = self._labels(rec)
            labels.append(lab)
            label_masks.append(mask)
            context.append(torch.tensor(active_window[t-1]['ids'], dtype=torch.long)
                           if t > 0 and self.args.history_reports and mode != 'remove_report' else None)
        record = dict(self._evaluation_records[idx], history_intervention=mode,
                      mechanism_intervention=intervention(self.args)['mechanism'])
        return images, images2, context, targets, masks, labels, label_masks, len(active_window), record
