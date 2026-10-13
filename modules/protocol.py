import csv
import hashlib
import json
import math
import posixpath
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from .contract import immutable_json

SPLITS = ('train', 'val', 'test')
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
               'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
               'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
               'Pneumothorax', 'Support Devices']

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def identifier(value, prefix=''):
    s = str(value).strip() if value is not None else ''
    if prefix and s.lower().startswith(prefix):
        s = s[len(prefix):]
    if s.endswith('.0') and s[:-2].isdigit():
        s = s[:-2]
    if not s.isdigit():
        raise ValueError('Missing or invalid numeric identifier: {!r}'.format(value))
    return str(int(s))

def image_path(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Missing image path')
    normalized = posixpath.normpath(value.strip().replace('\\', '/'))
    if normalized in ('.', '..') or normalized.startswith(('/', '../')) or re.match(r'^[A-Za-z]:', normalized):
        raise ValueError('Image path must remain relative to image_dir: '+value)
    return normalized

def training_labels(record, metadata=None, require=False):
    raw = record.get('label')
    if raw is None:
        raw = (metadata or {}).get('labels', {})
    if not isinstance(raw, (dict, list, tuple)):
        raise ValueError('Training labels must be a named dictionary or explicitly ordered list')
    if isinstance(raw, dict):
        unknown = set(raw)-set(CLASS_NAMES)
        if unknown or (raw and set(raw) != set(CLASS_NAMES)):
            raise ValueError('Training label dictionary must use all 14 CLASS_NAMES')
        values = [raw.get(c) for c in CLASS_NAMES]
    else:
        if len(raw) != 14 or record.get('label_names') != CLASS_NAMES:
            raise ValueError('List labels require explicit label_names matching CLASS_NAMES')
        values = raw
    labels, masks = [], []
    for value in values:
        if value is None or str(value).strip().lower() in ('', 'nan', 'none'):
            number = float('nan')
        else:
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError('Invalid CheXpert training label: {!r}'.format(value))
        if not math.isnan(number) and number not in (-1.0, 0.0, 1.0):
            raise ValueError('Training requires CheXpert -1/0/1 labels, not CheXbert class IDs: '+str(value))
        known = number in (0.0, 1.0)
        labels.append(number if known else 0.0)
        masks.append(float(known))
    if require and not raw:
        raise ValueError('Missing disease label record for training study '+str(record.get('study_id'))+
            '. Metadata timestamps are not disease labels. Connect the CheXpert CSV with attach_training_labels.py; '
            'do not fill missing labels with zero.')
    return labels, masks

def audit_annotation(ann):
    sets = {s: {'subject_id': set(), 'study_id': set(), 'image_path': set()} for s in SPLITS}
    study_owners = {}
    for split in SPLITS:
        if not isinstance(ann.get(split), list) or not ann[split]:
            raise ValueError('Missing or empty split: ' + split)
        for r in ann[split]:
            sub = identifier(r.get('subject_id'), 'p')
            study = identifier(r.get('study_id'), 's')
            if study in study_owners:
                raise ValueError('Duplicate study_id: ' + study)
            study_owners[study] = sub
            if r.get('split', split) != split:
                raise ValueError('Record split disagrees with enclosing split')
            paths = r.get('image_path')
            paths = paths if isinstance(paths, list) else [paths]
            if not paths or any(not isinstance(p, str) or not p.strip() for p in paths):
                raise ValueError('Missing image_path: ' + study)
            normalized_paths = {image_path(p) for p in paths}
            if sets[split]['image_path'] & normalized_paths:
                raise ValueError('Image reused by multiple studies: ' + study)
            if not isinstance(r.get('report'), str) or not r['report'].strip():
                raise ValueError('Empty report: ' + study)
            sets[split]['subject_id'].add(sub)
            sets[split]['study_id'].add(study)
            sets[split]['image_path'].update(normalized_paths)
    overlaps = {}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i+1:]:
            for key in sets[left]:
                shared = sets[left][key] & sets[right][key]
                overlaps[left + '__' + right + '__' + key] = len(shared)
                if shared:
                    raise ValueError('Split overlap {} / {} / {}: {}'.format(left, right, key, sorted(shared)[:5]))
    return {'subject_ids': {s: sorted(sets[s]['subject_id']) for s in SPLITS},
            'counts': {s: {k: len(v) for k, v in sets[s].items()} for s in SPLITS},
            'overlaps': overlaps}

def metadata_conflict(path, study, current, current_line, reason):
    fields = ('subject_id', 'study_id', 'dicom_id', 'StudyDate', 'StudyTime')
    first = None
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if identifier(row['study_id'], 's') == study:
                first = dict(csv_line=reader.line_num, **{k:row.get(k) for k in fields})
                break
    detail = {'metadata_path':str(Path(path).resolve()), 'reason':reason, 'study_id':study,
              'first_record':first,
              'conflicting_record':dict(csv_line=current_line, **{k:current.get(k) for k in fields})}
    return ValueError('Conflicting study metadata:\n'+json.dumps(detail,ensure_ascii=False,indent=2)+
        '\nTraining has not started. Check these source records; no rows have been removed or dates changed.')


def read_metadata(path):
    out = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not {'study_id', 'subject_id', 'StudyDate'}.issubset(reader.fieldnames or []):
            raise ValueError('Metadata requires subject_id, study_id, StudyDate columns')
        for row in reader:
            study = identifier(row['study_id'], 's')
            subject = identifier(row['subject_id'], 'p')
            date = row['StudyDate'].strip().split('.')[0]
            raw_time = row.get('StudyTime', '').strip()
            has_time = bool(raw_time and raw_time.lower() != 'nan')
            clock = raw_time.split('.')[0].zfill(6) if has_time else '000000'
            stamp = datetime.strptime(date + clock, '%Y%m%d%H%M%S')
            if has_time and '.' in raw_time:
                stamp = stamp.replace(microsecond=int((raw_time.split('.')[1] + '000000')[:6]))
            # Unknown time covers the whole day; preserve every recorded endpoint.
            end = stamp if has_time else stamp.replace(hour=23,minute=59,second=59,microsecond=999999)
            item = {'subject_id': subject, 'timestamp': stamp, 'timestamp_end':end,
                    'date_only': not has_time, 'metadata_rows':1,
                    'labels': {c: row[c] for c in CLASS_NAMES if c in row}}
            if study in out:
                old = out[study]
                if old['subject_id'] != subject:
                    raise metadata_conflict(path,study,row,reader.line_num,'same_study_multiple_subjects')
                # A calendar-date boundary alone does not prove an incorrect study ID.
                # Preserve the interval, including overnight records, without editing CSV.
                item['timestamp'] = min(old['timestamp'], stamp)
                item['timestamp_end'] = max(old['timestamp_end'],end)
                item['date_only'] = old['date_only'] or item['date_only']
                item['metadata_rows'] = old['metadata_rows']+1
                for c in item['labels']:
                    old_value, new_value = old['labels'].get(c, ''), item['labels'][c]
                    if old_value and new_value and old_value != new_value:
                        raise ValueError('Conflicting disease labels for study ' + study)
                    item['labels'][c] = old_value or new_value
            out[study] = item
    return out

def study_interval(item):
    start = item['timestamp']
    end = item.get('timestamp_end')
    if end is None:
        # Compatibility with fixtures using the earlier metadata shape.
        end = start
        if item.get('date_only',False):
            start = start.replace(hour=0,minute=0,second=0,microsecond=0)
            end = end.replace(hour=23,minute=59,second=59,microsecond=999999)
    return start,end


def metadata_time_summary(metadata):
    cross_date = []
    max_span = 0.0
    for study,item in metadata.items():
        start,end = study_interval(item)
        seconds = (end-start).total_seconds()
        max_span = max(max_span,seconds)
        if start.date()!=end.date():
            cross_date.append({'study_id':study,'subject_id':item['subject_id'],
                'start':start.isoformat(),'end':end.isoformat(),'span_seconds':seconds,
                'has_unknown_time':item['date_only']})
    cross_date.sort(key=lambda r:(-r['span_seconds'],r['study_id']))
    return {'studies':len(metadata),'multirow_studies':sum(m.get('metadata_rows',1)>1 for m in metadata.values()),
            'cross_date_studies':len(cross_date),'date_only_studies':sum(m['date_only'] for m in metadata.values()),
            'max_interval_seconds':max_span,'cross_date_examples_longest_first':cross_date[:20],
            'history_rule':'historical_study_end_strictly_before_current_study_start',
            'note':'Image-time intervals do not establish report signing/availability times; source records are unchanged.'}


def build_windows(records, metadata, max_history=3):
    if max_history < 0:
        raise ValueError('max_history must be nonnegative')
    patients = defaultdict(list)
    for raw in records:
        rec = dict(raw)
        sub = identifier(rec['subject_id'], 'p')
        study = identifier(rec['study_id'], 's')
        if study not in metadata or metadata[study]['subject_id'] != sub:
            raise ValueError('Missing metadata or subject mismatch for study: ' + study)
        rec.update(subject_id=sub, study_id=study)
        if 'image_path' in rec:
            paths = rec['image_path'] if isinstance(rec['image_path'], list) else [rec['image_path']]
            rec['image_path'] = [image_path(p) for p in paths]
        patients[sub].append(rec)
    windows, ambiguous = [], 0
    for sub in sorted(patients):
        seq = sorted(patients[sub], key=lambda r: (metadata[r['study_id']]['timestamp'], r['study_id']))
        for current in seq:
            cm = metadata[current['study_id']]
            current_start,current_end = study_interval(cm)
            previous = []
            for candidate in seq:
                if candidate is current:
                    continue
                hm = metadata[candidate['study_id']]
                history_start,history_end = study_interval(hm)
                if not (history_end < current_start or current_end < history_start):
                    ambiguous += 1
                    continue
                if history_end < current_start:
                    previous.append(candidate)
            history = previous[-max_history:] if max_history else []
            windows.append(history + [current])
    counts = Counter(len(w) for w in windows)
    visits = [len(s) for s in patients.values()]
    stats = {'targets': len(windows), 'subjects': len(patients),
             'sequence_lengths': dict(sorted(counts.items())),
             'with_history': sum(len(w) > 1 for w in windows),
             'ambiguous_ordered_pairs_excluded': ambiguous,
             'visits_mean': sum(visits)/len(visits) if visits else 0,
             'visits_min': min(visits) if visits else 0, 'visits_max': max(visits) if visits else 0,
             'visits_distribution': dict(sorted(Counter(visits).items()))}
    return windows, stats

def run_audit(ann_path, metadata_path, output_dir, max_history=3, image_dir=None,
              require_training_labels=False):
    ann = json.loads(Path(ann_path).read_text(encoding='utf-8'))
    audit = audit_annotation(ann)
    metadata = read_metadata(metadata_path)
    included_studies = {identifier(r['study_id'],'s') for split in SPLITS for r in ann[split]}
    audit['metadata_time_summary'] = metadata_time_summary({s:m for s,m in metadata.items() if s in included_studies})
    audit['windows'] = {s: build_windows(ann[s], metadata, max_history)[1] for s in SPLITS}
    label_support = [0] * 14
    supervised_studies = 0
    for r in ann['train']:
        _, mask = training_labels(r, metadata[identifier(r['study_id'], 's')], require_training_labels)
        supervised_studies += int(any(mask))
        label_support = [n+int(v) for n, v in zip(label_support, mask)]
    audit['training_supervision'] = {'known_labels_per_class': dict(zip(CLASS_NAMES, label_support)),
                                      'supervised_studies': supervised_studies,
                                      'studies_without_known_labels':len(ann['train'])-supervised_studies,
                                      'unknown_label_policy':'retain_report_loss_mask_auxiliary_classification'}
    if require_training_labels and not supervised_studies:
        raise ValueError('No known disease supervision in the entire training split. '
                         'Check annotation/label source before training a DGL model.')
    if image_dir is not None:
        resolved_owners = {}
        for split in SPLITS:
            for r in ann[split]:
                paths = r['image_path'] if isinstance(r['image_path'], list) else [r['image_path']]
                for p in paths:
                    resolved = (Path(image_dir) / image_path(p)).resolve(strict=True)
                    if not resolved.is_file():
                        raise FileNotFoundError(str(resolved))
                    owner = (split, identifier(r['study_id'], 's'))
                    if resolved in resolved_owners and resolved_owners[resolved] != owner:
                        raise ValueError('Same physical image assigned to multiple studies: '+str(resolved))
                    resolved_owners[resolved] = owner
    audit.update(annotation_sha256=sha256(ann_path), metadata_sha256=sha256(metadata_path),
                 max_history=max_history, images_checked=image_dir is not None)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    subjects = audit.pop('subject_ids')
    audit = json.loads(json.dumps(audit))
    immutable_json(output / 'split_subject_ids.json', subjects)
    immutable_json(output / 'split_audit.json', audit)
    return audit


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Read-only metadata check; no GPU or annotation generation.')
    parser.add_argument('--metadata_path', required=True)
    args = parser.parse_args()
    metadata = read_metadata(args.metadata_path)
    print(json.dumps({'metadata_path':str(Path(args.metadata_path).resolve()),
                      'studies':len(metadata), 'status':'metadata_check_passed',
                      'time_summary':metadata_time_summary(metadata)},indent=2))
