import os
import json
import pandas as pd
import random
import re
from pathlib import Path
from modules.contract import atomic_json
from collections import defaultdict

IMAGE_ROOT = 'data/mimic-cxr/images'
REPORT_ROOT = 'data/mimic-cxr/mimic-cxr-reports'
METADATA_CSV = 'mimic-cxr-2.0.0-metadata.csv'
OUTPUT_JSON = 'data/mimic-cxr/annotation.json'

split_ratio = {'train': 0.9, 'val': 0.05, 'test': 0.05}
SEED = 123
MAX_CHARS = 400
DUPLICATE_SINGLE_VIEW = True
MIN_STUDIES_PER_SUBJECT = 2

# cleaning thresholds
MIN_TARGET_CHARS = 30   
TEMPLATE_PHRASES = [
    r'no acute cardiopulmonary process',
    r'no acute cardiopulmonary disease',
    r'no acute cardiopulmonary abnormality',
    r'no acute cardiopulmonary findings'
]

RE_DUP_PAREN = re.compile(r'\s*\(\d+\)$')
RE_DUP_UND_HYPH = re.compile(r'[_-]\d+$')
RE_NONPRINT = re.compile(r'[^\x09\x0A\x0D\x20-\x7E\u4e00-\u9fff]')

RE_FINDINGS = re.compile(r'(?is)(?:\bfinding\b|\bfindings\b)\s*[:\-]?\s*(.*?)(?=\n[A-Z ]{2,30}:|\Z)')
RE_HEADER = re.compile(r'(?im)^(final report|report|history|indication|clinical history|exam)\b.*$', re.MULTILINE)
RE_SIGNATURE = re.compile(r'(?is)(?:signature:|dictated by|attending:|electronically signed by|performed by).*$')
RE_DATE = re.compile(r'\b(?:\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{1,2}:\d{2}(?::\d{2})?)\b')
RE_MULTI_SP = re.compile(r'[ \t]{2,}')

TEMPLATE_PATTERNS = [re.compile(p, re.I) for p in TEMPLATE_PHRASES]

def is_template_like(text: str) -> bool:
    if not text:
        return True
    t = text.strip().lower()
    if len(t) < MIN_TARGET_CHARS:
        return True
    for pat in TEMPLATE_PATTERNS:
        if pat.search(t):
            return True
    return False

def clean_report_extract_findings(raw: str):
    if not raw:
        return None
    text = RE_NONPRINT.sub(' ', raw.strip())
    m = RE_FINDINGS.search(text)
    if not m:
        return None
    fin = m.group(1).strip()
    # postprocess
    s = RE_HEADER.sub('', fin)
    s = RE_SIGNATURE.sub('', s)
    s = RE_DATE.sub('', s)
    s = re.sub(r'X{3,}', '[MASK]', s)
    s = re.sub(r'\n+', '. ', s)
    s = RE_MULTI_SP.sub(' ', s).strip()
    s = s.strip(' .')
    return s if s else None

def build_report_index(report_root):
    report_index = {}
    if not os.path.exists(report_root):
        print(f"[WARN] report_root 路径不存在：{report_root}")
        return report_index
    for root, _, files in os.walk(report_root):
        for file in files:
            lf = file.lower()
            if lf.endswith(".txt") and lf.startswith("s"):
                sid = file[1:-4]
                try:
                    study_id = int(sid)
                except ValueError:
                    continue
                if study_id in report_index:
                    raise ValueError('Duplicate report files for study: ' + str(study_id))
                report_index[study_id] = os.path.join(root, file)
    return report_index

def find_images_for_study(image_root, subject_id, study_id):
    pXX = f"p{str(subject_id)[:2]}"
    subj_folder = os.path.join(pXX, f"p{subject_id}")
    study_folder = f"s{study_id}"
    full_dir = os.path.join(image_root, subj_folder, study_folder)
    if not os.path.isdir(full_dir):
        return []
    files = [f for f in os.listdir(full_dir) if f.lower().endswith('.jpg')]
    if not files:
        return []
    files_sorted = sorted(files)[:2]
    image_paths = [os.path.join(pXX, f"p{subject_id}", study_folder, f).replace("\\", "/")
                   for f in files_sorted]
    if DUPLICATE_SINGLE_VIEW and len(image_paths) == 1:
        image_paths = [image_paths[0], image_paths[0]]
    return image_paths

def sanitize_dicom_id(raw_filename):
    name = os.path.splitext(os.path.basename(raw_filename))[0]
    name = name.strip()
    name = RE_NONPRINT.sub('', name)
    candidates = [name]
    no_paren = RE_DUP_PAREN.sub('', name).strip()
    if no_paren != name:
        candidates.append(no_paren)
    no_undhyp = RE_DUP_UND_HYPH.sub('', name).strip()
    if no_undhyp != name and no_undhyp != no_paren:
        candidates.append(no_undhyp)
    lc = name.lower()
    if lc not in candidates:
        candidates.append(lc)
    seen = set()
    uniq = []
    for c in candidates:
        if c and c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq

def main():
    if Path(OUTPUT_JSON).exists():
        raise FileExistsError('Refusing to overwrite: ' + str(OUTPUT_JSON))
    if not os.path.exists(METADATA_CSV):
        raise FileNotFoundError(f"找不到 metadata 文件: {METADATA_CSV}")

    df = pd.read_csv(METADATA_CSV, dtype=str, low_memory=False)
    meta_first_col_values = df.iloc[:, 0].astype(str).tolist()
    meta_index_set = set([v.strip() for v in meta_first_col_values])

    col_sub = None
    col_study = None
    for c in df.columns:
        cl = c.lower()
        if cl in ('subject_id', 'subjectid', 'patientid', 'pid'):
            col_sub = c
        if cl in ('study_id', 'studyid', 'studyinstanceuid'):
            col_study = c

    if col_sub is None or col_study is None:
        raise RuntimeError("metadata 文件中未找到 subject_id 或 study_id 列，请确认 CSV 的列名。")

    keep_cols = [col_sub, col_study]
    df_use = df[keep_cols].dropna().drop_duplicates().copy()
    df_use[col_sub] = df_use[col_sub].astype(int)
    df_use[col_study] = df_use[col_study].astype(int)

    report_index = build_report_index(REPORT_ROOT)
    print(f"[INFO] 找到 {len(report_index)} 份报告（txt 文件）。")
    print(f"[INFO] 仅保留含有效 FINDINGS（长度>={MIN_TARGET_CHARS}且非模板）的样本。")

    samples = []
    stats = {'total_seen_reports':0, 'kept_by_findings':0}

    for record_number, (_, row) in enumerate(df_use.iterrows(), 1):
        if record_number % 10000 == 0:
            print(f'[BUILD] scanned={record_number}, kept={len(samples)}', flush=True)
        subject_id = int(row[col_sub])
        study_id = int(row[col_study])

        report_path = report_index.get(study_id, None)
        if report_path is None or not os.path.exists(report_path):
            continue

        try:
            with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_report = f.read().strip()
        except Exception as e:
            print(f"[WARN] 无法读取报告 {report_path}: {e}")
            continue

        if not raw_report:
            continue

        stats['total_seen_reports'] += 1
        findings_text = clean_report_extract_findings(raw_report)
        if findings_text is None:
            continue
        if is_template_like(findings_text):
            continue

        target = findings_text
        if len(target) > MAX_CHARS:
            truncated = target[:MAX_CHARS]
            last_dot = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
            if last_dot > int(MAX_CHARS * 0.5):
                target = truncated[:last_dot+1]
            else:
                target = truncated
        target = target.strip()

        image_paths = find_images_for_study(IMAGE_ROOT, subject_id, study_id)
        if not image_paths:
            continue

        first_relpath = image_paths[0]
        candidates = sanitize_dicom_id(first_relpath)
        chosen = None
        for cid in candidates:
            if cid in meta_index_set:
                chosen = cid
                break
        if chosen is None:
            raw_name = os.path.splitext(os.path.basename(first_relpath))[0]
            if raw_name in meta_index_set:
                chosen = raw_name
        if chosen is None:
            continue

        stats['kept_by_findings'] += 1

        samples.append({
            "id": chosen,
            "study_id": int(study_id),
            "subject_id": int(subject_id),
            "report": target,
            "image_path": image_paths
        })

    print(f"[INFO] total_reports_seen={stats['total_seen_reports']}, kept_by_findings={stats['kept_by_findings']}")

    subj_to_samples = defaultdict(list)
    for s in samples:
        subj_to_samples[s['subject_id']].append(s)

    kept = []
    dropped_subj = 0
    for subj, recs in subj_to_samples.items():
        if len(recs) >= MIN_STUDIES_PER_SUBJECT:
            kept.extend(recs)
        else:
            dropped_subj += 1

    samples = kept
    print(f"[INFO] 丢弃 subject (<{MIN_STUDIES_PER_SUBJECT}) count={dropped_subj}, remaining_samples={len(samples)}")

    random.seed(SEED)
    subj_list = sorted({s['subject_id'] for s in samples})  # stable input order before seed=123 shuffle
    random.shuffle(subj_list)
    n_subj = len(subj_list)
    n_train = int(n_subj * split_ratio['train'])
    n_val = int(n_subj * split_ratio['val'])

    train_subjs = set(subj_list[:n_train])
    val_subjs = set(subj_list[n_train:n_train + n_val])
    test_subjs = set(subj_list[n_train + n_val:])

    annotation = {'train': [], 'val': [], 'test': []}
    for s in samples:
        dest = 'train' if s['subject_id'] in train_subjs else ('val' if s['subject_id'] in val_subjs else 'test')
        s_out = {
            "id": s['id'],
            "study_id": s['study_id'],
            "subject_id": s['subject_id'],
            "report": s['report'],
            "image_path": s['image_path'],
            "split": dest
        }
        annotation[dest].append(s_out)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    if any(not annotation[split] for split in ('train', 'val', 'test')):
        raise ValueError('Empty split after legacy filtering; check input completeness. No annotation saved.')
    atomic_json(OUTPUT_JSON, annotation)

    print(f"[SUCCESS] annotation.json 已保存至：{OUTPUT_JSON}")
    print({split: len(rows) for split, rows in annotation.items()})


if __name__ == '__main__':
    raise SystemExit('Use sbatch rebuild_annotation.sbatch to build, attach labels and audit together.')
