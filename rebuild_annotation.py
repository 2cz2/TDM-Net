import argparse
import json
import platform
from pathlib import Path

import generate_annotation as legacy
from attach_training_labels import attach_labels
from deployment import ROOT, read_json, verify_release
from modules.contract import atomic_json, immutable_json, run_lock, source_hash
from modules.protocol import run_audit, sha256


def verify_images(annotation, image_dir):
    from PIL import Image
    names = sorted({name for split in ('train', 'val', 'test')
                    for row in annotation[split] for name in row['image_path']})
    for number, name in enumerate(names, 1):
        with Image.open(Path(image_dir) / name) as image:
            image.load()
        if number % 10000 == 0:
            print('[IMAGES] decoded {}/{}'.format(number, len(names)), flush=True)
    return len(names)


def rebuild(p, max_history=3):
    for key in ('image_dir', 'report_dir'):
        if not Path(p[key]).is_dir():
            raise FileNotFoundError('{}: {}'.format(key, p[key]))
    for key in ('metadata_path', 'labels_csv'):
        if not Path(p[key]).is_file():
            raise FileNotFoundError('{}: {}'.format(key, p[key]))
    raw, labeled = Path(p['raw_annotation']), Path(p['ann_path'])
    if raw.resolve() == labeled.resolve():
        raise ValueError('Raw and labeled annotation must use different files')
    audit_dir = raw.parent / 'rebuild_audit'
    audit_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        'cohort': 'legacy_rules_rebuilt_patient_90_5_5_v1',
        'not_an_exact_recovery_of_previous_cohort': True,
        'split_seed': legacy.SEED, 'split_ratio': legacy.split_ratio,
        'patient_order_before_shuffle': 'sorted_numeric_subject_id',
        'min_studies_per_subject': legacy.MIN_STUDIES_PER_SUBJECT,
        'min_findings_chars': legacy.MIN_TARGET_CHARS,
        'max_report_chars': legacy.MAX_CHARS,
        'excluded_template_phrases': legacy.TEMPLATE_PHRASES,
        'image_selection': 'first_two_sorted_jpg_filenames_duplicate_single_view',
        'max_history': max_history,
        'image_dir': str(Path(p['image_dir']).resolve()),
        'report_dir': str(Path(p['report_dir']).resolve()),
        'metadata_sha256': sha256(p['metadata_path']),
        'labels_csv_sha256': sha256(p['labels_csv']),
        'generator_sha256': sha256(ROOT / 'generate_annotation.py'),
        'rebuild_script_sha256': sha256(ROOT / 'rebuild_annotation.py'),
        'label_script_sha256': sha256(ROOT / 'attach_training_labels.py'),
        'model_source_sha256': source_hash(),
        'python_version': platform.python_version(),
    }
    with run_lock(audit_dir):
        raw_receipt = audit_dir / 'raw_complete.json'
        label_receipt = audit_dir / 'labels_complete.json'
        for target, receipt in ((raw, raw_receipt), (labeled, label_receipt)):
            if target.exists() != receipt.exists():
                raise FileExistsError('Unpaired output/receipt; no file overwritten: {} / {}. '
                                      'Inspect the previous job before continuing.'.format(target, receipt))
        immutable_json(audit_dir / 'rebuild_contract.json', contract)
        if raw.exists():
            if read_json(raw_receipt)['sha256'] != sha256(raw):
                raise ValueError('Raw annotation changed since generation')
            print('[1/3] Reusing verified raw annotation', flush=True)
        else:
            print('[1/3] Building raw annotation with approved legacy rules', flush=True)
            legacy.IMAGE_ROOT = p['image_dir']
            legacy.REPORT_ROOT = p['report_dir']
            legacy.METADATA_CSV = p['metadata_path']
            legacy.OUTPUT_JSON = str(raw)
            legacy.main()
            atomic_json(raw_receipt, {'sha256': sha256(raw)})
        annotation = read_json(raw)
        if labeled.exists():
            receipt = read_json(label_receipt)
            if receipt['sha256'] != sha256(labeled) or receipt['input_sha256'] != sha256(raw):
                raise ValueError('Labeled annotation or its input changed')
            print('[2/3] Reusing verified labeled annotation', flush=True)
        else:
            print('[2/3] Attaching training-only CheXpert labels', flush=True)
            labeled_annotation, summary = attach_labels(annotation, p['labels_csv'])
            atomic_json(labeled, labeled_annotation)
            atomic_json(label_receipt, {'sha256': sha256(labeled),
                                       'input_sha256': sha256(raw), 'summary': summary})
        print('[3/3] Auditing splits, supervision, chronology and selected images', flush=True)
        audit = run_audit(str(labeled), p['metadata_path'], str(audit_dir),
                          max_history=max_history, image_dir=p['image_dir'],
                          require_training_labels=True)
        decoded = verify_images(annotation, p['image_dir'])
        complete = {'raw_annotation_sha256': sha256(raw),
                    'annotation_sha256': sha256(labeled),
                    'unique_images_decoded': decoded, 'counts': audit['counts'],
                    'training_supervision': audit['training_supervision']}
        immutable_json(audit_dir / 'rebuild_complete.json', complete)
        print(json.dumps(complete, indent=2), flush=True)
        print('REBUILD COMPLETE: ' + str(labeled), flush=True)
        print('No training started. Keep both annotations and the entire rebuild_audit folder.', flush=True)
    return complete


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', default=str(ROOT / 'config/paths.json'))
    args = parser.parse_args()
    verify_release()
    configured = read_json(args.paths)
    resolved = {key: str((ROOT / Path(value)).resolve()) for key, value in configured.items()}
    rebuild(resolved, read_json(ROOT / 'config/study.json')['max_history'])
