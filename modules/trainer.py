"""Validation-only selection, study-ID outputs and explicit final evaluation."""
import inspect
import json
import random
import time
import os
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .protocol import sha256
from .contract import (PROTOCOL, TRAINING_KEYS, DECODE_KEYS, configuration,
    source_hash, json_hash, tensor_hash, intervention, condition_name, validate_intervention,
    immutable_json, atomic_json, atomic_text)

def save_checkpoint(state, path):
    path = Path(path)
    fd, temp = tempfile.mkstemp(prefix='.'+path.name+'.', dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(state, temp)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)

def load_checkpoint(path, device):
    options = {'map_location': device}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = False
    checkpoint = torch.load(path, **options)
    if checkpoint.get('protocol') != PROTOCOL:
        raise ValueError('Old or unaudited checkpoint: training must restart under the revision protocol.')
    return checkpoint

class Trainer:
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler,
                 train_dataloader, val_dataloader, test_dataloader):
        self.args, self.model = args, model
        requested = getattr(args, 'device', 'auto')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if requested=='auto' else torch.device(requested)
        self.model.to(self.device)
        self.criterion, self.metric_ftns = criterion, metric_ftns
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.train_loader, self.val_loader, self.test_loader = train_dataloader, val_dataloader, test_dataloader
        self.out = Path(args.save_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.best, self.start_epoch, self.not_improved = float('-inf'), 1, 0
        self.audit = json.loads((self.out/'split_audit.json').read_text(encoding='utf-8'))
        self.code_sha256 = source_hash()
        self.training_settings = configuration(args, TRAINING_KEYS)
        self.decoding_settings = configuration(args, DECODE_KEYS)
        if args.resume:
            if (self.out/'test_selection.json').exists():
                raise ValueError('Training cannot resume after test selection has been locked')
            state = self._load(args.resume)
            self.optimizer.load_state_dict(state['optimizer'])
            self.lr_scheduler.load_state_dict(state['scheduler'])
            self.best, self.start_epoch = state['best_val'], state['epoch']+1
            self.not_improved = state['not_improved']
            random.setstate(state['python_rng'])
            np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng'].cpu())
            if torch.cuda.is_available() and state['cuda_rng'] is not None:
                torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])

    def _load(self, path):
        state = load_checkpoint(path, self.device)
        for key in ('annotation_sha256', 'metadata_sha256'):
            if state['audit'][key] != self.audit[key]:
                raise ValueError('Checkpoint data fingerprint mismatch: ' + key)
        if state.get('code_sha256') != self.code_sha256:
            raise ValueError('Model source changed since training; the checkpoint cannot be mixed with this code')
        for name, settings in (('training_settings',self.training_settings),('decoding_settings',self.decoding_settings)):
            if state.get(name) != settings:
                changed = [key for key in settings if state.get(name,{}).get(key)!=settings[key]]
                raise ValueError('Checkpoint configuration mismatch: '+', '.join(changed))
        self.model.load_state_dict(state['state_dict'])
        self.loaded_checkpoint_sha256 = sha256(path)
        return state

    def _batch(self, batch):
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def _current(self, batch, field):
        value = batch[field]
        return value[torch.arange(value.size(0), device=self.device), batch['seq_len']-1]

    def _forward(self, batch, mode):
        return self.model(images=batch['images_seq'], images2=batch['images2_seq'],
                          context=batch['context_seq'], time_lengths=batch['seq_len'],
                          targets=self._current(batch, 'targets') if mode == 'train' else None,
                          mode=mode)

    def _decode(self, values):
        tokenizer = self.model.tokenizer
        texts = []
        for seq in values.detach().cpu().tolist():
            ids = []
            for token in seq:
                if token == tokenizer.eos_id:
                    break
                if token not in (tokenizer.bos_id, tokenizer.pad_id):
                    ids.append(token)
            texts.append(tokenizer.decode(ids))
        return texts

    def generation_contract(self):
        return {'protocol': PROTOCOL, 'generation_code_sha256': self.code_sha256,
                'checkpoint_sha256': getattr(self,'loaded_checkpoint_sha256',None),
                'annotation_sha256': self.audit['annotation_sha256'],
                'metadata_sha256': self.audit['metadata_sha256'],
                'training_settings_sha256': json_hash(self.training_settings),
                'decoding': self.decoding_settings}

    def _generate(self, loader):
        self.model.eval()
        records, seen = [], set()
        with torch.no_grad():
            for raw in loader:
                batch = self._batch(raw)
                reports = self._decode(self._forward(batch, 'sample'))
                gates = self.model.encoder_decoder.last_history_gate.flatten().cpu().tolist()
                learned = self.model.encoder_decoder.last_learned_history_gate.flatten().cpu().tolist()
                memory = self.model.encoder_decoder.last_memory_gate_mean.flatten().cpu().tolist()
                if not len(reports)==len(gates)==len(raw['records']):
                    raise ValueError('Prediction batch size mismatch')
                for i,(record, prediction, gate) in enumerate(zip(raw['records'], reports, gates)):
                    key = (record['subject_id'], record['study_id'])
                    if key in seen:
                        raise ValueError('Duplicate prediction ID: ' + str(key))
                    seen.add(key)
                    current_index = int(raw['seq_len'][i]) - 1
                    current_hash = tensor_hash(raw['images_seq'][i,current_index],
                        raw['images2_seq'][i,current_index] if raw['images2_seq'] is not None else None)
                    records.append(dict(record, prediction=prediction, history_gate=gate,
                        learned_history_gate=learned[i], memory_gate_mean=memory[i],
                        current_input_sha256=current_hash, protocol=PROTOCOL,
                        generation_contract=self.generation_contract(), intervention=intervention(self.args),
                        history_intervention=intervention(self.args)['history'],
                        mechanism_intervention=intervention(self.args)['mechanism'],
                        generation_checkpoint_sha256=getattr(self, 'loaded_checkpoint_sha256', None),
                        experiment=self.args.experiment, seed=self.args.seed))
        if not records or len(records) != len(loader.dataset):
            raise ValueError('Evaluation did not cover every target study')
        self._verify_records(records, loader)
        return records

    def _verify_records(self, records, loader):
        if not records or len(records)!=len(loader.dataset):
            raise ValueError('Prediction coverage mismatch')
        keys = [(r['subject_id'],r['study_id']) for r in records]
        if len(set(keys))!=len(keys):
            raise ValueError('Duplicate prediction IDs')
        expected_contract, expected_intervention = self.generation_contract(), intervention(self.args)
        for r in records:
            if r.get('generation_contract')!=expected_contract or r.get('intervention')!=expected_intervention:
                raise ValueError('Existing predictions have a different model, decoding setting or intervention')
        if hasattr(loader.dataset,'evaluation_records'):
            expected={(r['subject_id'],r['study_id']):r for r in loader.dataset.evaluation_records()}
            if set(keys)!=set(expected):
                raise ValueError('Predictions do not cover the exact target study ID set')
            for r in records:
                for key,value in expected[(r['subject_id'],r['study_id'])].items():
                    if r.get(key)!=value:
                        raise ValueError('Target metadata mismatch: '+key)

    def _score(self, records):
        clean = self.model.tokenizer.clean_report
        return self.metric_ftns({i: [clean(r['reference'])] for i,r in enumerate(records)},
                               {i: [clean(r['prediction'])] for i,r in enumerate(records)})

    def evaluate(self, loader, filename):
        records=self._generate(loader)
        metrics=self._score(records)
        atomic_text(self.out/filename,''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),overwrite=True)
        return metrics

    def train(self):
        validate_intervention(self.args,training=True)
        if (self.out/'test_selection.json').exists():
            raise ValueError('This experiment already reached test selection; training is locked')
        log_path = self.out/'training_log.jsonl'
        for epoch in range(self.start_epoch, self.args.epochs+1):
            began = time.monotonic()
            self.model.train()
            # Fixed pretrained BN statistics prevent padding/history composition from changing the current anchor.
            for module in self.model.visual_extractor.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
            total, steps = 0.0, 0
            known_disease_labels, batches_without_disease_labels = 0, 0
            for raw in self.train_loader:
                batch = self._batch(raw)
                output, cls_logits = self._forward(batch, 'train')
                targets = self._current(batch, 'targets')[:, 1:]
                masks = self._current(batch, 'targets_masks')[:, 1:]
                loss = self.criterion(output, targets, masks)
                if self.args.use_dgl:
                    labels = self._current(batch, 'labels')
                    label_masks = self._current(batch, 'label_masks')
                    # Explicitly unknown labels do not become negatives or remove reports.
                    # The preflight audit separately rejects an entirely unsupervised DGL run.
                    known = int(label_masks.sum().item())
                    known_disease_labels += known
                    batches_without_disease_labels += int(known == 0)
                    cls_loss = F.binary_cross_entropy_with_logits(cls_logits, labels, reduction='none',
                            pos_weight=torch.full_like(labels[0], self.args.cls_pos_weight))
                    loss = loss + self.args.cls_lambda*(cls_loss*label_masks).sum()/label_masks.sum().clamp(min=1)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                total += loss.item()
                steps += 1
                if steps % 100 == 0:
                    print('epoch={} batch={}/{} loss={:.5f}'.format(epoch,steps,len(self.train_loader),total/steps), flush=True)
            if not steps:
                raise ValueError('Empty training loader')
            val = self.evaluate(self.val_loader, 'validation_latest.jsonl')
            score = val[self.args.monitor_metric]
            if not np.isfinite(score):
                raise FloatingPointError('Non-finite validation metric')
            improved = score > self.best
            self.not_improved = 0 if improved else self.not_improved+1
            self.best = max(self.best, score)
            self.lr_scheduler.step(score)
            log = {'epoch': epoch, 'train_loss': total/steps, 'validation': val,
                   'selected': improved, 'elapsed_seconds': time.monotonic()-began}
            if self.args.use_dgl:
                log.update(known_disease_labels=known_disease_labels,
                           batches_without_disease_labels=batches_without_disease_labels)
            with log_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(log)+'\n')
            print(json.dumps(log), flush=True)
            state = {'protocol': PROTOCOL, 'epoch': epoch, 'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(), 'scheduler': self.lr_scheduler.state_dict(),
                     'best_val': self.best, 'not_improved': self.not_improved,
                     'args': dict(vars(self.args)), 'audit': self.audit, 'python_rng': random.getstate(),
                     'code_sha256': self.code_sha256, 'training_settings':self.training_settings,
                     'decoding_settings':self.decoding_settings,
                     'is_validation_selected': improved,
                     'selection': {'split':'val','metric':self.args.monitor_metric,'score':score,'epoch':epoch},
                     'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
            save_checkpoint(state, self.out/'current_checkpoint.pth')
            if improved:
                save_checkpoint(state, self.out/'model_best.pth')
            if self.not_improved >= self.args.early_stop:
                break
        # No test evaluation during model selection. Test is a separate explicit run.
        if not (self.out/'model_best.pth').exists():
            raise RuntimeError('Training finished without a validation-selected checkpoint')
        atomic_json(self.out/'training_complete.json',
            {'protocol':PROTOCOL,'checkpoint_sha256':sha256(self.out/'model_best.pth'),
             'best_val':self.best,'training_settings':self.training_settings},overwrite=True)

    def test(self):
        validate_intervention(self.args,training=False)
        if not self.args.checkpoint:
            raise ValueError('--mode test requires --checkpoint selected on validation')
        state=self._load(self.args.checkpoint)
        if not state.get('is_validation_selected') or state.get('selection',{}).get('split')!='val':
            raise ValueError('Testing requires a validation-selected checkpoint, not an arbitrary epoch')
        training_receipt = json.loads((Path(self.args.checkpoint).parent/'training_complete.json').read_text(encoding='utf-8'))
        if training_receipt.get('checkpoint_sha256') != self.loaded_checkpoint_sha256:
            raise ValueError('Test checkpoint is not the final validation winner of the completed training run')
        selected={'generation_contract':self.generation_contract(),'selection':state['selection']}
        immutable_json(self.out/'test_selection.json',selected)
        name = 'test_' + condition_name(self.args)
        prediction_path=self.out/(name+'.jsonl')
        completion_path=self.out/(name+'_complete.json')
        generation_path=self.out/(name+'_generated.json')
        if completion_path.exists():
            raise FileExistsError('Completed test results are locked: '+name)
        if prediction_path.exists():
            if not getattr(self.args,'retry_scoring',False):
                raise FileExistsError('Predictions already exist; use --retry_scoring to recover scoring without regenerating')
            records=[json.loads(line) for line in prediction_path.read_text(encoding='utf-8').splitlines()]
            receipt=json.loads(generation_path.read_text(encoding='utf-8'))
            if receipt.get('prediction_sha256') != sha256(prediction_path):
                raise ValueError('Saved predictions changed after generation')
            self._verify_records(records,self.test_loader)
        else:
            records=self._generate(self.test_loader)
            atomic_text(prediction_path,''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
            atomic_json(generation_path,{'prediction_sha256':sha256(prediction_path),'n_records':len(records)})
        result=self._score(records)
        immutable_json(self.out/(name+'_nlg.json'),result)
        atomic_json(completion_path,{'protocol':PROTOCOL,'prediction_sha256':sha256(prediction_path),
                    'n_records':len(records),'generation_contract':self.generation_contract(),
                    'intervention':intervention(self.args)})
        print(json.dumps(result), flush=True)
