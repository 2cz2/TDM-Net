import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Enlarged Cardiomediastinum',
    'Fracture',
    'Lung Lesion',
    'Lung Opacity',
    'No Finding',
    'Pleural Effusion',
    'Pleural Other',
    'Pneumonia',
    'Pneumothorax',
    'Support Devices'
]


class BaseDataset(Dataset):

    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform

        with open(self.ann_path, 'r') as f:
            self.ann = json.loads(f.read())
        examples_raw = self.ann[self.split]

        metadata = pd.read_csv(args.metadata_path, index_col=0)
        if "study_id" in metadata.columns:
            metadata["study_id"] = metadata["study_id"].astype(str)
            metadata.set_index("study_id", inplace=True)
        metadata.index = metadata.index.astype(str)

        keep_cols = ['StudyDate']
        for c in CLASS_NAMES:
            if c in metadata.columns:
                keep_cols.append(c)

        self.metadata = (
            metadata[keep_cols]
            if set(keep_cols).issubset(metadata.columns)
            else metadata
        )

        for ex in examples_raw:
            ex['ids'] = tokenizer(ex['report'])[:self.max_seq_length]
            ex['mask'] = [1] * len(ex['ids'])

            study_id = ex['id'].lstrip('s')
            try:
                study_date = self.metadata.loc[study_id]['StudyDate']
                if isinstance(study_date, pd.Series):
                    study_date = study_date.values[0]
            except Exception:
                study_date = '0'

            ex['time'] = str(study_date)

        df = pd.DataFrame(examples_raw)
        df.sort_values(['subject_id', 'time'], inplace=True)
        records = df.to_dict('records')

        subs = defaultdict(list)
        for rec in records:
            subs[rec['subject_id']].append(rec)

        self.examples = list(subs.values())

    def __len__(self):
        return len(self.examples)

    def _parse_labels(self, raw_values):
        labels = [0.0] * len(CLASS_NAMES)
        label_mask = [0.0] * len(CLASS_NAMES)

        if raw_values is None:
            return labels, label_mask

        for idx, v in enumerate(raw_values):
            if pd.isna(v):
                labels[idx] = 0.0
                label_mask[idx] = 0.0
            elif float(v) == -1.0:
                labels[idx] = 0.0
                label_mask[idx] = 0.0
            else:
                labels[idx] = float(v)
                label_mask[idx] = 1.0

        return labels, label_mask


class MimiccxrSequenceDataset(BaseDataset):

    def __getitem__(self, idx):
        patient_seq = self.examples[idx]

        MAX_HISTORY = 4
        if len(patient_seq) > MAX_HISTORY:
            patient_seq = patient_seq[-MAX_HISTORY:]

        images_list, images2_list = [], []
        context_list, targets_list, masks_list = [], [], []
        labels_list, label_masks_list = [], []

        for i, rec in enumerate(patient_seq):
            img_paths = rec['image_path']
            if isinstance(img_paths, list) and len(img_paths) >= 1:
                primary_path = img_paths[0]
                secondary_path = img_paths[1] if len(img_paths) > 1 else None
            else:
                primary_path = img_paths
                secondary_path = None

            try:
                img_primary = Image.open(
                    os.path.join(self.image_dir, primary_path)
                ).convert('RGB')
            except FileNotFoundError:
                img_primary = Image.new('RGB', (224, 224), (0, 0, 0))

            if self.transform:
                img_primary = self.transform(img_primary)

            images_list.append(img_primary)

            if secondary_path is not None:
                try:
                    img_secondary = Image.open(
                        os.path.join(self.image_dir, secondary_path)
                    ).convert('RGB')
                except FileNotFoundError:
                    img_secondary = Image.new('RGB', (224, 224), (0, 0, 0))

                if self.transform:
                    img_secondary = self.transform(img_secondary)

                images2_list.append(img_secondary)
            else:
                images2_list.append(None)

            targets_list.append(torch.LongTensor(rec['ids']))
            masks_list.append(torch.LongTensor(rec['mask']))

            labels = [0.0] * len(CLASS_NAMES)
            label_mask = [0.0] * len(CLASS_NAMES)

            if 'label' in rec and rec['label'] is not None:
                labels, label_mask = self._parse_labels(rec['label'])
            else:
                study_id = rec['id'].lstrip('s')
                if study_id in self.metadata.index:
                    row = self.metadata.loc[study_id]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    raw_vals = [
                        row.get(c, np.nan) if c in row.index else np.nan
                        for c in CLASS_NAMES
                    ]
                    labels, label_mask = self._parse_labels(raw_vals)

            labels_list.append(torch.FloatTensor(labels))
            label_masks_list.append(torch.FloatTensor(label_mask))

            if i == 0:
                context_list.append(None)
            else:
                prev_ids = patient_seq[i - 1].get('ids', None)
                if prev_ids is not None:
                    context_list.append(torch.LongTensor(prev_ids))
                else:
                    context_list.append(None)

        seq_len = len(images_list)

        return (
            images_list,
            images2_list,
            context_list,
            targets_list,
            masks_list,
            labels_list,
            label_masks_list,
            seq_len
        )