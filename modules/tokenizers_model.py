import json
import re
from collections import Counter


class Tokenizer(object):

    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        else:
            self.clean_report = self.clean_report_mimic_cxr
            
        with open(self.ann_path, 'r') as f:
            self.ann = json.loads(f.read())

        self.token2idx, self.idx2token = self.create_vocabulary()

        self.pad_id = self.token2idx['<pad>']
        self.bos_id = self.token2idx['<bos>']
        self.eos_id = self.token2idx['<eos>']
        self.unk_id = self.token2idx['<unk>']

    def create_vocabulary(self):
        total_tokens = []
        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            total_tokens.extend(tokens)

        counter = Counter(total_tokens)
        special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']

        vocab_candidates = [k for k, v in counter.items() if v >= self.threshold]
        vocab_candidates = [w for w in sorted(vocab_candidates) if w not in special_tokens]
        vocab = special_tokens + vocab_candidates

        token2idx = {token: idx for idx, token in enumerate(vocab)}
        idx2token = {idx: token for idx, token in enumerate(vocab)}
        
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: (
            t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '')
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ')
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ')
            .strip().lower().split('. ')
        )
        sent_cleaner = lambda t: re.sub(
            '[.,?;*!%^&_+():-\[\]{}]', '', 
            t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()
        )
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        return ' . '.join(tokens) + ' .'

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: (
            t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_')
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ')
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.')
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ')
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ')
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ')
            .strip().lower().split('. ')
        )
        sent_cleaner = lambda t: re.sub(
            '[.,?;*!%^&_+():-\[\]{}]', '', 
            t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()
        )
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        return ' . '.join(tokens) + ' .'

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        return self.token2idx.get(token, self.unk_id)

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = [self.get_id_by_token(token) for token in tokens]
        return [self.bos_id] + ids + [self.eos_id]

    def decode(self, ids):
        words = []
        for idx in ids:
            idx = int(idx)
            if idx == self.eos_id:
                break
            if idx in (self.pad_id, self.bos_id):
                continue
            words.append(self.idx2token.get(idx, '<unk>'))
        return ' '.join(words)

    def decode_batch(self, ids_batch):
        return [self.decode(ids) for ids in ids_batch]