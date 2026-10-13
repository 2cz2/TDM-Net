import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def paths():
    return {k: str((ROOT / Path(v)).resolve())
            for k, v in read_json(ROOT / 'config/paths.json').items()}


def setup_environment():
    os.environ['TORCH_HOME'] = paths()['torch_home']
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['USE_TF'] = '0'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def verify_release():
    manifest = read_json(ROOT / 'release_integrity.json')
    bad = [name for name, expected in manifest['python_files'].items()
           if not (ROOT/name).is_file() or digest(ROOT/name) != expected]
    if bad:
        raise ValueError('Release source modified/incomplete: ' + ', '.join(bad))
    from modules.contract import source_hash
    if source_hash() != manifest['core_sha256']:
        raise ValueError('Model source file set differs from the actual trained snapshot')


def arguments(values):
    derived = {'protocol', 'vocab_size', 'history_images', 'history_reports',
               'use_cmi', 'use_te', 'use_mg', 'use_dgl'}
    argv = []
    for key, value in values.items():
        if key in derived or value is None:
            continue
        if key in ('prompt_detach', 'retry_scoring'):
            if value:
                argv.append('--' + key)
        else:
            argv.extend(['--' + key, str(value)])
    return argv


def run_directory(experiment):
    study = read_json(ROOT/'config/study.json')
    return Path(paths()['runs_root']) / experiment / ('seed_' + str(study['seed']))


def training_values(experiment):
    p = paths()
    values = read_json(ROOT/'config/study.json')
    if experiment in ('current_mg_dgl', 'history_image_only', 'direct_concat', 'longitudinal_mg', 'history_report_only', 'longitudinal'):
        values['early_stop'] = 5
    return dict(values, experiment=experiment,
                image_dir=p['image_dir'], ann_path=p['ann_path'], metadata_path=p['metadata_path'],
                save_dir=str(run_directory(experiment)), mode='train',
                history_intervention='factual', mechanism_intervention='none')


def restored_values(run, output, history='factual', mechanism='none'):
    p = paths()
    return dict(read_json(Path(run)/'run_config.json'), mode='test',
                checkpoint=str(Path(run)/'model_best.pth'), resume=None, retry_scoring=False,
                save_dir=str(output), image_dir=p['image_dir'], ann_path=p['ann_path'],
                metadata_path=p['metadata_path'], history_intervention=history,
                mechanism_intervention=mechanism, device=read_json(ROOT/'config/study.json')['device'])


def execute(command):
    logdir = Path(paths()['runs_root']) / 'job_logs'
    logdir.mkdir(parents=True, exist_ok=True)
    logfile = logdir / ('%s_%s_%s.log' % (time.strftime('%Y%m%d_%H%M%S'), os.getpid(), time.time_ns()))
    print('RUN: ' + ' '.join(str(x) for x in command), flush=True)
    print('LOG: ' + str(logfile), flush=True)
    with logfile.open('x', encoding='utf-8') as log:
        with subprocess.Popen([str(x) for x in command], cwd=str(ROOT), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as proc:
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            status = proc.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)


def python(script, *argv):
    execute([sys.executable, '-u', str(ROOT/script), *map(str, argv)])


def require_data():
    p = paths()
    for key in ('ann_path', 'metadata_path'):
        if not Path(p[key]).is_file():
            raise FileNotFoundError(key + ': ' + p[key])
    if not Path(p['image_dir']).is_dir():
        raise FileNotFoundError('image_dir: ' + p['image_dir'])


def require_resnet():
    import torchvision.models as models
    if hasattr(models, 'ResNet101_Weights'):
        url = models.ResNet101_Weights.IMAGENET1K_V1.url
    else:
        from torchvision.models.resnet import model_urls
        url = model_urls['resnet101']
    name = url.rsplit('/', 1)[-1]
    target = Path(paths()['torch_home'])/'hub/checkpoints'/name
    if not target.is_file():
        raise FileNotFoundError('Put ImageNet ResNet101 V1 weights at ' + str(target) +
                                '. Official URL reported by torchvision: ' + url)
    prefix = name.rsplit('-', 1)[-1].split('.')[0]
    value = digest(target)
    if not value.startswith(prefix):
        raise ValueError('ResNet checkpoint checksum mismatch: ' + str(target))
    return {'file': name, 'sha256': value}


def require_ce():
    p = paths()
    for target in (Path(p['chexbert']), Path(p['bert'])/'config.json', Path(p['bert'])/'vocab.txt'):
        if not target.is_file():
            raise FileNotFoundError('CE requires: ' + str(target))
    if not any((Path(p['bert'])/name).is_file() for name in
               ('pytorch_model.bin', 'model.safetensors', 'pytorch_model.bin.index.json', 'model.safetensors.index.json')):
        raise FileNotFoundError('BERT base weights are missing in ' + p['bert'])


def verify_completed(run):
    receipt = read_json(Path(run)/'training_complete.json')
    if digest(Path(run)/'model_best.pth') != receipt['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from training completion receipt')
    return receipt


def check_prediction(path):
    path = Path(path)
    receipt = read_json(path.with_name(path.stem + '_complete.json'))
    if digest(path) != receipt['prediction_sha256']:
        raise ValueError('Prediction checksum mismatch: ' + str(path))
    with path.open(encoding='utf-8') as f:
        if sum(1 for line in f if line.strip()) != receipt['n_records']:
            raise ValueError('Prediction count mismatch: ' + str(path))
    return receipt
