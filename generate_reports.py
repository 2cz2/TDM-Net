"""Generate validation/test reports from the completed validation-selected model only."""
import argparse
import json
from pathlib import Path
import random
import time
from deployment import (ROOT, arguments, restored_values, read_json, digest, paths,
                        setup_environment, verify_release, verify_completed, check_prediction)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_dir', type=Path, required=True)
    p.add_argument('--split', choices=['val', 'test'], required=True)
    p.add_argument('--history', choices=['factual', 'none', 'remove_image', 'remove_report'], default='factual')
    p.add_argument('--mechanism', choices=['none', 'gate_closed', 'gate_open', 'prompt_off'], default='none')
    cli = p.parse_args()
    setup_environment()
    verify_release()
    if cli.split == 'val' and (cli.history != 'factual' or cli.mechanism != 'none'):
        p.error('This release uses validation only for the predeclared factual model.')
    run = cli.run_dir.resolve()
    receipt = verify_completed(run)
    if cli.split == 'test':
        lock = read_json(Path(paths()['runs_root'])/'evaluation_lock.json')
        if lock['study'] != read_json(ROOT/'config/study.json'):
            raise ValueError('Study settings changed after evaluation lock')
    import numpy as np
    import torch
    from main import parse_agrs
    from models.r2gen import R2GenModel
    from modules.dataloaders import R2DataLoader
    from modules.loss import compute_loss
    from modules.metrics import compute_scores
    from modules.contract import (PROTOCOL, atomic_text, atomic_json, immutable_json,
                                           run_lock, source_hash, condition_name)
    from modules.protocol import run_audit
    from modules.trainer import Trainer
    from modules.tokenizers_model import Tokenizer
    out = run/'evaluation'/cli.split
    args = parse_agrs(arguments(restored_values(run, out, cli.history, cli.mechanism)))
    if cli.split == 'test':
        frozen = lock['contract']
        if (source_hash() != frozen['source_sha256'] or
                digest(args.ann_path) != frozen['annotation_sha256'] or
                digest(args.metadata_path) != frozen['metadata_sha256']):
            raise ValueError('Test source/data differ from the locked study')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use the scheduled GPU job')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    name = ('validation_' if cli.split == 'val' else 'test_') + condition_name(args)
    prediction = out/(name+'.jsonl')
    with run_lock(run), run_lock(out):
        audit = run_audit(args.ann_path, args.metadata_path, out, args.max_history,
                          image_dir=args.image_dir, require_training_labels=False)
        if audit != read_json(run/'split_audit.json'):
            raise ValueError('Data audit differs from the trained run')
        tokenizer = Tokenizer(args)
        if tokenizer.token2idx != read_json(run/'vocabulary.json'):
            raise ValueError('Vocabulary differs from training')
        args.vocab_size = tokenizer.get_vocab_size()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        model = R2GenModel(args, tokenizer).to(device)
        trainer = Trainer(model, compute_loss, compute_scores, None, args, None, None, None, None)
        state = trainer._load(args.checkpoint)
        if (not state.get('is_validation_selected') or state['selection']['split'] != 'val'
                or state['selection']['score'] != receipt['best_val']):
            raise ValueError('Checkpoint is not the final validation winner')
        if cli.split == 'test':
            immutable_json(run/'test_selection.json',
                           {'generation_contract': trainer.generation_contract(), 'selection': state['selection']})
        loader = R2DataLoader(args, tokenizer, cli.split, False)
        completion = out/(name+'_complete.json')
        if completion.exists():
            check_prediction(prediction)
            rows = [json.loads(line) for line in prediction.read_text(encoding='utf-8').splitlines()]
            trainer._verify_records(rows, loader)
            print('Verified completed prediction: ' + str(prediction), flush=True)
            return
        generated = out/(name+'_generated.json')
        if prediction.exists():
            if digest(prediction) != read_json(generated)['prediction_sha256']:
                raise ValueError('Incomplete prediction checksum mismatch')
            rows = [json.loads(line) for line in prediction.read_text(encoding='utf-8').splitlines()]
            trainer._verify_records(rows, loader)
            print('Reusing generated reports; retrying NLG scoring only.', flush=True)
        else:
            start = time.monotonic()
            rows = trainer._generate(loader)
            elapsed = time.monotonic() - start
            for row in rows:
                row.update(evaluation_split=cli.split, checkpoint_epoch=state['epoch'])
            atomic_text(prediction, ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
            atomic_json(generated, {'split': cli.split, 'prediction_sha256': digest(prediction),
                                    'n_records': len(rows), 'generation_seconds': elapsed,
                                    'timing_scope': 'end-to-end one pass including data loading; not a warmed latency benchmark'})
        scores = trainer._score(rows)
        if cli.split == 'val' and abs(scores[args.monitor_metric]-receipt['best_val']) > 1e-5:
            raise ValueError('Regenerated validation score differs from training best; inspect data/environment before testing')
        immutable_json(out/(name+'_nlg.json'), scores)
        atomic_json(completion, {'protocol': PROTOCOL, 'split': cli.split,
                                 'prediction_sha256': digest(prediction), 'n_records': len(rows),
                                 'generation_contract': trainer.generation_contract(),
                                 'intervention': {'history': cli.history, 'mechanism': cli.mechanism},
                                 'checkpoint_epoch': state['epoch']})
        print(json.dumps({'split': cli.split, 'epoch': state['epoch'], 'nlg': scores}, indent=2), flush=True)


if __name__ == '__main__':
    main()
