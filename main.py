import sys
import os
import torch
import argparse
import random
import json
from pathlib import Path
from modules.protocol import run_audit
from modules.contract import (PROTOCOL, HISTORY_CONDITIONS, MECHANISM_CONDITIONS,
    validate_intervention, immutable_json, run_lock)
from modules.config import EXPERIMENTS, configure
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycocoevalcap', 'pycocoevalcap'))

from modules.tokenizers_model import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.optimizers import build_optimizer
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.r2gen import R2GenModel


def parse_agrs(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', type=str, default='data/mimic_cxr/images/')
    parser.add_argument('--ann_path', type=str, default='data/mimic_cxr/annotation.json')
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr', choices=['mimic_cxr'])
    parser.add_argument('--max_seq_length', type=int, default=60)
    parser.add_argument('--threshold', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--visual_extractor', type=str, default='resnet101')
    parser.add_argument('--visual_extractor_pretrained', type=int, choices=[0, 1], default=1)

    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--d_vf', type=int, default=2048)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)

    parser.add_argument('--max_time_steps', type=int, default=50)
    parser.add_argument('--num_temporal_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--logit_layers', type=int, default=1)
    parser.add_argument('--bos_idx', type=int, default=1)
    parser.add_argument('--eos_idx', type=int, default=2)
    parser.add_argument('--pad_idx', type=int, default=0)
    parser.add_argument('--use_bn', type=int, default=0)
    parser.add_argument('--drop_prob_lm', type=float, default=0.5)

    parser.add_argument('--rm_num_slots', type=int, default=3)
    parser.add_argument('--rm_num_heads', type=int, default=8)
    parser.add_argument('--rm_d_model', type=int, default=512)

    parser.add_argument('--sample_method', type=str, default='greedy')
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--length_alpha', type=float, default=1.3)
    parser.add_argument('--repetition_penalty', type=float, default=1.05)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=3)

    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sample_n', type=int, default=1)
    parser.add_argument('--group_size', type=int, default=1)
    parser.add_argument('--output_logsoftmax', type=int, default=1)
    parser.add_argument('--decoding_constraint', type=int, default=0)
    parser.add_argument('--block_trigrams', type=int, default=1)

    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='result/mimic_cxr')
    parser.add_argument('--record_dir', type=str, default='records/')
    parser.add_argument('--save_period', type=int, default=1)
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'])
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4')
    parser.add_argument('--early_stop', type=int, default=50)

    parser.add_argument('--optim', choices=['Adam','AdamW'], default='Adam')
    parser.add_argument('--lr_ve', type=float, default=5e-5)
    parser.add_argument('--lr_ed', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--amsgrad', type=int, choices=[0, 1], default=1)

    parser.add_argument('--lr_scheduler', type=str, default='StepLR')
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.1)

    parser.add_argument('--seed', type=int, default=9233)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--metadata_path', type=str, default='mimic-cxr-2.0.0-metadata.csv')

    parser.add_argument('--experiment', choices=tuple(EXPERIMENTS), default='full')
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--checkpoint')
    parser.add_argument('--max_history', type=int, default=3)
    parser.add_argument('--max_gen_length', type=int, default=100)
    parser.add_argument('--cls_lambda', type=float, default=2.0)
    parser.add_argument('--cls_pos_weight', type=float, default=2.5)
    parser.add_argument('--prompt_detach', action='store_true')
    parser.add_argument('--history_dropout', type=float, default=0.15)
    parser.add_argument('--history_intervention', choices=['factual', 'none', 'remove_image', 'remove_report'], default='factual')
    parser.add_argument('--mechanism_intervention', choices=MECHANISM_CONDITIONS, default='none')
    parser.add_argument('--retry_scoring', action='store_true')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    args = parser.parse_args(argv)
    configure(args)
    try:
        validate_intervention(args,training=args.mode=='train')
    except ValueError as e:
        parser.error(str(e))
    if args.mode=='test' and (not args.checkpoint or args.resume):
        parser.error('Test requires --checkpoint and cannot use --resume')
    if args.mode=='train' and (args.checkpoint or args.retry_scoring):
        parser.error('--checkpoint and --retry_scoring are test-only options')
    if args.max_history < 0 or args.max_history >= args.max_time_steps:
        parser.error('max_history must be nonnegative and less than max_time_steps')
    if args.max_seq_length < 3 or min(args.max_gen_length,args.batch_size,args.beam_size,args.epochs,args.early_stop) < 1:
        parser.error('Invalid sequence length, batch size, beam size, epochs or early stop')
    if args.num_workers < 0 or args.length_alpha < 0 or args.repetition_penalty < 1 or args.no_repeat_ngram_size < 0:
        parser.error('Invalid loader or decoding parameters')
    if args.use_bn != 0:
        parser.error('use_bn must be 0 for the revision feature projection')
    if min(args.d_model,args.d_ff,args.d_vf,args.num_heads,args.num_layers,args.rm_d_model,args.rm_num_heads,args.rm_num_slots,args.num_temporal_layers)<1:
        parser.error('Model dimensions, head counts and layer counts must be positive')
    if args.d_model % args.num_heads or args.rm_d_model % args.rm_num_heads:
        parser.error('Embedding dimensions must be divisible by the number of attention heads')
    if (args.pad_idx,args.bos_idx,args.eos_idx) != (0,1,2):
        parser.error('Special token IDs must match the frozen tokenizer: PAD=0 BOS=1 EOS=2')
    if args.monitor_mode != 'max':
        parser.error('Revision uses maximization of a predeclared validation metric')
    if args.mode == 'train' and args.history_intervention != 'factual':
        parser.error('Interventions are for locked-checkpoint testing only')
    if not 0 <= args.history_dropout <= 1:
        parser.error('history_dropout must be in [0,1]')
    # Legacy flags are retained for readable migration errors, never silently applied.
    legacy = {'sample_method':'greedy','temperature':1.0,'sample_n':1,'group_size':1,
              'output_logsoftmax':1,'decoding_constraint':0,'block_trigrams':1,
              'lr_scheduler':'StepLR','step_size':50,'gamma':0.1,'logit_layers':1,'save_period':1}
    if any(getattr(args,k)!=v for k,v in legacy.items()):
        parser.error('Legacy decoding/scheduler options are inactive. Use beam_size, no_repeat_ngram_size; revision scheduler is fixed ReduceLROnPlateau.')
    return args


def main():
    args = parse_agrs()
    from modules.metrics import compute_scores
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else torch.device(args.device)
    out = Path(args.save_dir)
    with run_lock(out):
        if args.mode=='train' and (out/'run_config.json').exists() and not args.resume:
            raise FileExistsError('Run already exists. Use a new directory or resume its revision checkpoint.')
        if args.mode=='train' and (out/'test_selection.json').exists():
            raise ValueError('Cannot train after the test checkpoint has been locked')
        if args.resume and Path(args.resume).resolve().parent != out.resolve():
            raise ValueError('Resume in the original run directory so the earlier best checkpoint is retained')
        run_audit(args.ann_path,args.metadata_path,out,args.max_history,image_dir=args.image_dir,
                  require_training_labels=args.mode=='train' and args.use_dgl)
        tokenizer = Tokenizer(args)
        immutable_json(out/'vocabulary.json',tokenizer.token2idx)
        args.vocab_size = tokenizer.get_vocab_size()
        if args.beam_size > args.vocab_size-2:
            raise ValueError('beam_size exceeds available vocabulary tokens')
        if args.mode=='train' and not args.resume:
            immutable_json(out/'run_config.json',dict(vars(args),protocol=PROTOCOL))
        train_loader = R2DataLoader(args,tokenizer,'train',True) if args.mode=='train' else None
        val_loader = R2DataLoader(args,tokenizer,'val',False) if args.mode=='train' else None
        test_loader = R2DataLoader(args,tokenizer,'test',False) if args.mode=='test' else None
        model = R2GenModel(args,tokenizer).to(device)
        optimizer = build_optimizer(args,model) if args.mode=='train' else None
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.5,
                     patience=5,min_lr=1e-6) if optimizer is not None else None
        trainer = Trainer(model,compute_loss,compute_scores,optimizer,args,scheduler,train_loader,val_loader,test_loader)
        trainer.train() if args.mode=='train' else trainer.test()


if __name__ == '__main__':
    main()
