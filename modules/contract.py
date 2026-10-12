"""Versioned contracts for locked models, paired interventions and durable artifacts."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile

PROTOCOL = 'tdm_revision_20260912_v2'
HISTORY_CONDITIONS = ('factual', 'none', 'remove_image', 'remove_report')
MECHANISM_CONDITIONS = ('none', 'gate_closed', 'gate_open', 'prompt_off')
TRAINING_KEYS = (
    'experiment', 'dataset_name', 'max_history', 'max_seq_length', 'threshold',
    'd_model', 'd_ff', 'd_vf', 'num_layers', 'num_heads', 'dropout', 'drop_prob_lm',
    'max_time_steps', 'num_temporal_layers', 'rm_num_slots', 'rm_d_model', 'rm_num_heads',
    'bos_idx', 'eos_idx', 'pad_idx', 'use_bn', 'visual_extractor',
    'visual_extractor_pretrained', 'prompt_detach', 'cls_lambda', 'cls_pos_weight',
    'history_dropout', 'seed', 'batch_size', 'num_workers', 'optim', 'lr_ve', 'lr_ed',
    'weight_decay', 'amsgrad', 'monitor_metric', 'early_stop', 'epochs',
)
DECODE_KEYS = ('max_gen_length', 'beam_size', 'length_alpha', 'repetition_penalty',
               'no_repeat_ngram_size')

def configuration(args, keys):
    return {key: getattr(args, key, None) for key in keys}

def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode('utf-8')).hexdigest()

def source_hash():
    root = Path(__file__).resolve().parents[1]
    files = [root/'main.py']
    for folder in ('models', 'modules'):
        files.extend((root/folder).glob('*.py'))
    actual = json_hash({p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(files)})
    # A rename release preserves the trained implementation identity only when
    # every physical core file still matches the audited normalized snapshot.
    migration_path = root / 'config' / 'source_migration.json'
    if migration_path.is_file():
        migration = json.loads(migration_path.read_text(encoding='utf-8'))
        if actual == migration.get('normalized_core_sha256'):
            return migration['original_core_sha256']
    return actual


def implementation_hash(path):
    """Retain cached label provenance for a verified import-only rename."""
    path = Path(path).resolve()
    root = Path(__file__).resolve().parents[1]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    migration_path = root / 'config' / 'source_migration.json'
    if migration_path.is_file() and path.parent == root:
        migration = json.loads(migration_path.read_text(encoding='utf-8'))
        entry = migration.get('renamed_implementations', {}).get(path.name, {})
        if actual == entry.get('normalized_sha256'):
            return entry['original_sha256']
    return actual

def intervention(args):
    return {'history': getattr(args, 'history_intervention', 'factual'),
            'mechanism': getattr(args, 'mechanism_intervention', 'none')}

def condition_name(args):
    value = intervention(args)
    return value['history'] if value['mechanism'] == 'none' else value['history']+'__'+value['mechanism']

def validate_intervention(args, training=False):
    value = intervention(args)
    if value['history'] not in HISTORY_CONDITIONS or value['mechanism'] not in MECHANISM_CONDITIONS:
        raise ValueError('Unknown intervention')
    if training and value != {'history': 'factual', 'mechanism': 'none'}:
        raise ValueError('Interventions are only allowed on a locked model during evaluation')
    if value['history'] != 'factual' and value['mechanism'] != 'none':
        raise ValueError('Do not combine history removal and a mechanism intervention in one contrast')
    if value['mechanism'] in ('gate_closed', 'gate_open'):
        if not getattr(args, 'use_mg', False) or not (args.history_images or args.history_reports):
            raise ValueError('History-gate intervention requires a model with MG and historical inputs')
    if value['mechanism'] == 'prompt_off' and not getattr(args, 'use_dgl', False):
        raise ValueError('Prompt intervention requires a model trained with DGL')
    return value

def atomic_text(path, content, overwrite=False):
    """Write a complete file before publishing it; refuse accidental overwrites."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.'+target.name+'.', dir=str(target.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        if overwrite:
            os.replace(temp, target)
        else:
            # Link publication is exclusive. Both paths are in the same filesystem.
            os.link(temp, target)
    finally:
        Path(temp).unlink(missing_ok=True)

def atomic_json(path, value, overwrite=False):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', overwrite)

def immutable_json(path, value):
    target = Path(path)
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8')) != value:
            raise ValueError('Frozen artifact differs from this run: '+str(target))
    else:
        atomic_json(target, value)

@contextmanager
def run_lock(directory):
    path = Path(directory)/'.run.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        f.write(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink(missing_ok=True)

def tensor_hash(*tensors):
    h = hashlib.sha256()
    for tensor in tensors:
        if tensor is None:
            h.update(b'None')
        else:
            cpu = tensor.detach().cpu().contiguous()
            h.update(str((str(cpu.dtype), tuple(cpu.shape))).encode('ascii'))
            h.update(cpu.numpy().tobytes())
    return h.hexdigest()
