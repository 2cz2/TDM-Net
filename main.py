import sys
import os
import torch
import argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycocoevalcap', 'pycocoevalcap'))

from modules.tokenizers_model import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.r2gen import R2GenModel


def parse_agrs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json')
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'])
    parser.add_argument('--max_seq_length', type=int, default=60)
    parser.add_argument('--threshold', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--visual_extractor', type=str, default='resnet101')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True)

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

    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr_ve', type=float, default=5e-5)
    parser.add_argument('--lr_ed', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--amsgrad', type=bool, default=True)

    parser.add_argument('--lr_scheduler', type=str, default='StepLR')
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.1)

    parser.add_argument('--seed', type=int, default=9233)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--metadata_path', type=str, default='mimic-cxr-2.0.0-metadata.csv')

    args = parser.parse_args()
    return args


def main():
    args = parse_agrs()

    print("正在使用的数据集是：", args.dataset_name)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    tokenizer = Tokenizer(args)

    if hasattr(tokenizer, 'get_vocab_size'):
        real_vocab_size = tokenizer.get_vocab_size()
    elif hasattr(tokenizer, 'idx2word'):
        real_vocab_size = len(tokenizer.idx2word)
    elif hasattr(tokenizer, 'word2idx'):
        real_vocab_size = len(tokenizer.word2idx)
    else:
        try:
            real_vocab_size = len(tokenizer)
        except TypeError:
            raise AttributeError("Tokenizer Error: Cannot determine vocab size!")

    args.vocab_size = real_vocab_size

    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    model = R2GenModel(args, tokenizer)

    criterion = compute_loss
    metrics = compute_scores
    optimizer = build_optimizer(args, model)
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )

    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()