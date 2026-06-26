import os
import json
from collections import defaultdict, Counter

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
    """
    Study-level longitudinal dataset.

    Main change compared with the previous version:
    - The previous version grouped all studies of one subject into one dataset item.
      Therefore, len(dataset) equaled the number of subjects.
    - This version uses each current study/report as one dataset item, while
      attaching its recent history as the input sequence.
      Therefore, len(dataset) is close to the number of studies/reports in the split.

    Each item is a sequence:
        [history_1, ..., history_k, current_study]
    where current_study is the prediction target at the last time step.
    """

    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform

        self.max_history = int(getattr(args, "max_history", 4))
        self.require_history = bool(getattr(args, "require_history", False))
        self.return_meta = bool(getattr(args, "return_meta", False))
        self.debug_dataset = bool(getattr(args, "debug_dataset", True))

        with open(self.ann_path, 'r', encoding='utf-8') as f:
            self.ann = json.load(f)

        if self.split not in self.ann:
            raise KeyError(f"Split '{self.split}' is not found in annotation file: {self.ann_path}")

        examples_raw = self.ann[self.split]
        self.metadata = self._load_metadata(args)

        processed_records = []
        for ex in examples_raw:
            rec = dict(ex)

            report = rec.get('report', '')
            rec['ids'] = tokenizer(report)[:self.max_seq_length]
            rec['mask'] = [1] * len(rec['ids'])

            study_key = self._study_key(rec)
            study_date, study_time = self._get_study_datetime(study_key)
            rec['StudyDate'] = str(study_date)
            rec['StudyTime'] = str(study_time)
            rec['time_key'] = self._make_time_key(study_date, study_time, rec.get('study_id', ''))

            processed_records.append(rec)

        df = pd.DataFrame(processed_records)
        df.sort_values(['subject_id', 'time_key', 'study_id'], inplace=True)
        records = df.to_dict('records')

        subs = defaultdict(list)
        for rec in records:
            subs[rec['subject_id']].append(rec)

        self.subject_sequences = dict(subs)

        self.examples = []
        for subject_id, seq in self.subject_sequences.items():
            seq = sorted(
                seq,
                key=lambda x: (str(x.get('time_key', '0')), str(x.get('study_id', '')))
            )

            for cur_idx in range(len(seq)):
                start_idx = max(0, cur_idx - self.max_history + 1)
                cur_seq = seq[start_idx:cur_idx + 1]

                if self.require_history and len(cur_seq) < 2:
                    continue

                self.examples.append(cur_seq)

        if self.debug_dataset:
            seq_len_counter = Counter(len(x) for x in self.examples)
            print("=" * 80)
            print(f"[Dataset Debug] split={self.split}")
            print(f"[Dataset Debug] raw studies/reports={len(examples_raw)}")
            print(f"[Dataset Debug] unique subjects={len(self.subject_sequences)}")
            print(f"[Dataset Debug] constructed study-level examples={len(self.examples)}")
            print(f"[Dataset Debug] max_history={self.max_history}")
            print(f"[Dataset Debug] require_history={self.require_history}")
            print(f"[Dataset Debug] sequence length distribution={seq_len_counter.most_common(20)}")

    def _load_metadata(self, args):
        metadata_path = getattr(args, "metadata_path", None)

        if metadata_path is None or not os.path.exists(metadata_path):
            print(f"[Dataset Warning] metadata_path is missing or not found: {metadata_path}")
            return pd.DataFrame()

        metadata = pd.read_csv(metadata_path)

        unnamed_cols = [c for c in metadata.columns if str(c).startswith("Unnamed")]
        if unnamed_cols:
            metadata = metadata.drop(columns=unnamed_cols)

        if "study_id" in metadata.columns:
            metadata["study_id"] = metadata["study_id"].astype(str).str.lstrip("s")
            metadata.set_index("study_id", inplace=True)
        else:
            metadata.index = metadata.index.astype(str).str.lstrip("s")

        keep_cols = []
        for col in ['StudyDate', 'StudyTime']:
            if col in metadata.columns:
                keep_cols.append(col)

        for c in CLASS_NAMES:
            if c in metadata.columns:
                keep_cols.append(c)

        if keep_cols:
            metadata = metadata[keep_cols]

        metadata.index = metadata.index.astype(str).str.lstrip("s")
        return metadata

    @staticmethod
    def _make_time_key(study_date, study_time, study_id):
        date = "0" if pd.isna(study_date) else str(study_date)
        time = "0" if pd.isna(study_time) else str(study_time)
        date = date.replace(".0", "")
        time = time.replace(".0", "")
        return f"{date}_{time}_{study_id}"

    @staticmethod
    def _to_scalar(value, default="0"):
        if isinstance(value, pd.Series):
            if len(value) == 0:
                return default
            return value.iloc[0]
        if pd.isna(value):
            return default
        return value

    @staticmethod
    def _study_key(rec):
        study_id = rec.get('study_id', rec.get('id', ''))
        return str(study_id).lstrip('s')

    def _get_metadata_row(self, study_key):
        if self.metadata.empty or study_key not in self.metadata.index:
            return None

        row = self.metadata.loc[study_key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row

    def _get_study_datetime(self, study_key):
        row = self._get_metadata_row(study_key)
        if row is None:
            return "0", "0"

        study_date = row.get('StudyDate', "0") if 'StudyDate' in row.index else "0"
        study_time = row.get('StudyTime', "0") if 'StudyTime' in row.index else "0"

        study_date = self._to_scalar(study_date, default="0")
        study_time = self._to_scalar(study_time, default="0")

        return study_date, study_time

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

    def _get_labels_for_record(self, rec):
        labels = [0.0] * len(CLASS_NAMES)
        label_mask = [0.0] * len(CLASS_NAMES)

        if 'label' in rec and rec['label'] is not None:
            return self._parse_labels(rec['label'])

        study_key = self._study_key(rec)
        row = self._get_metadata_row(study_key)
        if row is None:
            return labels, label_mask

        raw_vals = [
            row.get(c, np.nan) if c in row.index else np.nan
            for c in CLASS_NAMES
        ]
        return self._parse_labels(raw_vals)

    def _build_meta(self, patient_seq):
        cur = patient_seq[-1]
        prev = patient_seq[-2] if len(patient_seq) >= 2 else None

        return {
            "id": cur.get("id", None),
            "study_id": cur.get("study_id", None),
            "subject_id": cur.get("subject_id", None),
            "gt_report": cur.get("report", ""),
            "prev_id": prev.get("id", None) if prev is not None else None,
            "prev_study_id": prev.get("study_id", None) if prev is not None else None,
            "prev_report": prev.get("report", "") if prev is not None else "",
            "seq_len": len(patient_seq),
            "seq_study_ids": [x.get("study_id", None) for x in patient_seq],
            "seq_ids": [x.get("id", None) for x in patient_seq],
        }


class MimiccxrSequenceDataset(BaseDataset):

    def __getitem__(self, idx):
        patient_seq = self.examples[idx]

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
            except (FileNotFoundError, TypeError):
                img_primary = Image.new('RGB', (224, 224), (0, 0, 0))

            if self.transform:
                img_primary = self.transform(img_primary)

            images_list.append(img_primary)

            if secondary_path is not None:
                try:
                    img_secondary = Image.open(
                        os.path.join(self.image_dir, secondary_path)
                    ).convert('RGB')
                except (FileNotFoundError, TypeError):
                    img_secondary = Image.new('RGB', (224, 224), (0, 0, 0))

                if self.transform:
                    img_secondary = self.transform(img_secondary)

                images2_list.append(img_secondary)
            else:
                images2_list.append(None)

            targets_list.append(torch.LongTensor(rec['ids']))
            masks_list.append(torch.LongTensor(rec['mask']))

            labels, label_mask = self._get_labels_for_record(rec)
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

        output = (
            images_list,
            images2_list,
            context_list,
            targets_list,
            masks_list,
            labels_list,
            label_masks_list,
            seq_len
        )

        if self.return_meta:
            meta = self._build_meta(patient_seq)
            return output + (meta,)

        return output
