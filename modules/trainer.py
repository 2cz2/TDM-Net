import os
import time
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter


class BaseTrainer(object):

    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model.to(self.device)
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period
        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric

        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)
        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {
            'val': {self.mnt_metric: self.mnt_best},
            'test': {self.mnt_metric_test: self.mnt_best},
        }

        log_dir = os.path.join(self.checkpoint_dir, 'runs')
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0

        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            if epoch % 1 == 0:
                log = {'epoch': epoch}
                log.update(result)
                self._record_best(log)

                for key, value in log.items():
                    print('\t{:15s}: {}'.format(str(key), value))
                    if isinstance(value, (int, float)):
                        if 'val_' in str(key):
                            self.writer.add_scalar(
                                f'Val/{str(key).replace("val_", "")}', value, epoch
                            )
                        elif 'test_' in str(key):
                            self.writer.add_scalar(
                                f'Test/{str(key).replace("test_", "")}', value, epoch
                            )
                        elif key == 'train_loss':
                            self.writer.add_scalar('Train/Epoch_Loss', value, epoch)

            torch.cuda.empty_cache()

            best = False
            if self.mnt_mode != 'off':
                try:
                    improved = (
                        (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or
                        (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    )
                except KeyError:
                    print("Warning: Metric '{}' is not found.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn't improve for {} epochs. Training stops.".format(
                        self.early_stop
                    ))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

        self.writer.close()
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)

        record_path = os.path.join(self.args.record_dir, self.args.dataset_name + '.csv')

        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)

        record_table = pd.concat(
            [record_table, pd.DataFrame([self.best_recorder['val']])], ignore_index=True
        )
        record_table = pd.concat(
            [record_table, pd.DataFrame([self.best_recorder['test']])], ignore_index=True
        )
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()

        if n_gpu_use > 0 and n_gpu == 0:
            print('Warning: No GPU available, training will run on CPU.')
            n_gpu_use = 0

        if n_gpu_use > n_gpu:
            print(f'Warning: Configured to use {n_gpu_use} GPUs, but only {n_gpu} are available.')
            n_gpu_use = n_gpu

        device = torch.device('cuda' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
        }

        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)

        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print('Saving current best: model_best.pth ...')

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print('Loading checkpoint: {} ...'.format(resume_path))

        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print('Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (
            (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][self.mnt_metric]) or
            (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        )
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (
            (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][self.mnt_metric_test]) or
            (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][self.mnt_metric_test])
        )
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best result (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best result (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, 
                 train_dataloader, val_dataloader, test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.cls_pos_weight = torch.ones([14], device=self.device) * getattr(args, 'cls_pos_weight', 2.5)
        self.cls_lambda = getattr(args, 'cls_lambda', 2.0)
        
    def _masked_cls_loss(self, logits, labels, label_masks):
        if logits is None or labels is None:
            return torch.tensor(0.0, device=self.device)

        if label_masks is None:
            return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=self.cls_pos_weight)

        loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=self.cls_pos_weight, reduction='none'
        )
        loss = loss * label_masks
        denom = label_masks.sum().clamp(min=1.0)
        return loss.sum() / denom

    def _extract_target_for_timestep(self, targets, masks, labels, label_masks, seq_len):
        B = targets.size(0)
        is_training = self.model.training

        real_targets, real_masks, real_labels, real_label_masks, final_seq_lens = [], [], [], [], []
        T_list = seq_len.cpu().tolist() if isinstance(seq_len, torch.Tensor) else [targets.size(1)] * B

        for i in range(B):
            max_t = T_list[i]
            if is_training and max_t > 1:
                curr_t = np.random.randint(1, max_t)
            else:
                curr_t = max_t - 1

            final_seq_lens.append(curr_t + 1)
            curr_t = max(0, min(curr_t, targets.size(1) - 1))

            real_targets.append(targets[i, curr_t])
            if masks is not None:
                real_masks.append(masks[i, curr_t])
            if labels is not None:
                real_labels.append(labels[i, curr_t])
            if label_masks is not None:
                real_label_masks.append(label_masks[i, curr_t])

        out_targets = torch.stack(real_targets)
        out_masks = torch.stack(real_masks) if masks is not None else None
        out_labels = torch.stack(real_labels) if labels is not None and len(real_labels) > 0 else None
        out_label_masks = torch.stack(real_label_masks) if label_masks is not None and len(real_label_masks) > 0 else None
        out_seq_len = torch.tensor(final_seq_lens, device=targets.device)

        return out_targets, out_masks, out_labels, out_label_masks, out_seq_len

    def _clean_and_decode(self, token_ids_tensor):
        ids_list = token_ids_tensor.cpu().numpy().tolist() if isinstance(token_ids_tensor, torch.Tensor) else token_ids_tensor
        tokenizer = self.model.tokenizer
        
        PAD = getattr(tokenizer, 'pad_id', 0)
        BOS = getattr(tokenizer, 'bos_id', 1)
        EOS = getattr(tokenizer, 'eos_id', 2)

        cleaned_texts = []
        for seq in ids_list:
            clean_ids = []
            for x in seq:
                x = int(x)
                if x == EOS:
                    break
                if x in (PAD, BOS):
                    continue
                clean_ids.append(x)
            cleaned_texts.append(tokenizer.decode(clean_ids))

        return cleaned_texts

    def _move_optional_to_device(self, x):
        return x.to(self.device) if x is not None else None

    def _slice_optional(self, x, max_active_len):
        return x[:, :max_active_len].contiguous() if x is not None else None

    def _train_epoch(self, epoch):
        train_loss = 0.0
        current_lr = self.optimizer.param_groups[0]['lr']

        print(f'Begin train epoch {epoch}')
        print(f'Threshold = {self.model.encoder_decoder.clinical_prompt_threshold}')
        print(f'Cls Lambda = {self.cls_lambda}')
        print(f'Current Learning Rate: {current_lr:.10f}')

        self.model.train()
        for batch_idx, batch in enumerate(self.train_dataloader):
            self.global_step += 1

            if not isinstance(batch, dict):
                raise RuntimeError('collate_fn dictionary issue')

            images_seq = batch.get('images_seq').to(self.device)
            images2_seq = self._move_optional_to_device(batch.get('images2_seq', None))
            context_seq = self._move_optional_to_device(batch.get('context_seq', None))
            reports_ids = batch.get('targets').to(self.device)
            reports_masks = batch.get('targets_masks').to(self.device)
            labels_seq = self._move_optional_to_device(batch.get('labels', None))
            label_masks_seq = self._move_optional_to_device(batch.get('label_masks', None))
            seq_len = batch.get('seq_len').to(self.device)

            (
                real_targets, real_masks, real_labels, real_label_masks, active_seq_len
            ) = self._extract_target_for_timestep(
                reports_ids, reports_masks, labels_seq, label_masks_seq, seq_len
            )

            max_active_len = active_seq_len.max().item()
            images_seq_sliced = self._slice_optional(images_seq, max_active_len)
            images2_seq_sliced = self._slice_optional(images2_seq, max_active_len)
            context_seq_sliced = self._slice_optional(context_seq, max_active_len)

            output = self.model.forward_mimic_cxr(
                images=images_seq_sliced,
                context=context_seq_sliced,
                images2=images2_seq_sliced,
                targets=real_targets,
                labels=real_labels,
                time_lengths=active_seq_len,
                mode='train',
            )

            logits_gen, logits_cls = output if isinstance(output, tuple) else (output, None)

            B_tgt, L_tgt = real_targets.shape
            logits_gen_slice = logits_gen[:, :-1, :] if logits_gen.size(1) == L_tgt else logits_gen

            targets_slice = real_targets[:, 1:]
            masks_slice = real_masks[:, 1:] if real_masks is not None else torch.ones_like(targets_slice)

            loss_gen = self.criterion(logits_gen_slice, targets_slice, masks_slice)
            loss_cls = self._masked_cls_loss(logits_cls, real_labels, real_label_masks)
            loss = loss_gen + self.cls_lambda * loss_cls

            train_loss += loss.item()

            self.writer.add_scalar('Train/Step_Loss', loss.item(), self.global_step)
            self.writer.add_scalar('Train/Step_Loss_Gen', loss_gen.item(), self.global_step)
            self.writer.add_scalar('Train/Step_Loss_Cls', float(loss_cls.item()), self.global_step)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

        train_loss = train_loss / max(1, len(self.train_dataloader))
        log = {'train_loss': train_loss}

        print('Begin validation evaluation')
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []

            for batch_idx, batch in enumerate(self.val_dataloader):
                if not isinstance(batch, dict):
                    raise RuntimeError('collate_fn dictionary issue')

                images_seq = batch.get('images_seq').to(self.device)
                images2_seq = self._move_optional_to_device(batch.get('images2_seq', None))
                context_seq = self._move_optional_to_device(batch.get('context_seq', None))
                reports_ids = batch.get('targets').to(self.device)
                seq_len = batch.get('seq_len', None).to(self.device)

                real_targets, _, _, _, active_seq_len = self._extract_target_for_timestep(
                    reports_ids, None, None, None, seq_len
                )

                max_active_len = active_seq_len.max().item()

                output = self.model.forward_mimic_cxr(
                    images=self._slice_optional(images_seq, max_active_len),
                    context=self._slice_optional(context_seq, max_active_len),
                    images2=self._slice_optional(images2_seq, max_active_len),
                    targets=None,
                    labels=None,
                    time_lengths=active_seq_len,
                    mode='sample',
                )

                reports = self._clean_and_decode(output)
                ground_truths = self._clean_and_decode(real_targets[:, 1:])

                val_res.extend(reports)
                val_gts.extend(ground_truths)

            if len(val_gts) > 0:
                os.makedirs('result', exist_ok=True)
                pd.DataFrame(val_res).to_csv(f'result/{epoch}-res_val.csv', index=False, header=False)
                pd.DataFrame(val_gts).to_csv(f'result/{epoch}-gt_val.csv', index=False, header=False)
                print('Saved Validation CSVs to result/ folder')

            val_met = self.metric_ftns(
                {i: [gt] for i, gt in enumerate(val_gts)},
                {i: [re] for i, re in enumerate(val_res)},
            )
            log.update({('val_' + k): v for k, v in val_met.items()})

        print('Begin test evaluation')
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []

            for batch_idx, batch in enumerate(self.test_dataloader):
                if not isinstance(batch, dict):
                    raise RuntimeError('collate_fn dictionary issue')

                images_seq = batch.get('images_seq').to(self.device)
                images2_seq = self._move_optional_to_device(batch.get('images2_seq', None))
                context_seq = self._move_optional_to_device(batch.get('context_seq', None))
                reports_ids = batch.get('targets').to(self.device)
                seq_len = batch.get('seq_len', None).to(self.device)

                real_targets, _, _, _, active_seq_len = self._extract_target_for_timestep(
                    reports_ids, None, None, None, seq_len
                )

                max_active_len = active_seq_len.max().item()

                output = self.model.forward_mimic_cxr(
                    images=self._slice_optional(images_seq, max_active_len),
                    context=self._slice_optional(context_seq, max_active_len),
                    images2=self._slice_optional(images2_seq, max_active_len),
                    targets=None,
                    labels=None,
                    time_lengths=active_seq_len,
                    mode='sample',
                )

                reports = self._clean_and_decode(output)
                ground_truths = self._clean_and_decode(real_targets[:, 1:])

                test_res.extend(reports)
                test_gts.extend(ground_truths)

            if len(test_gts) > 0:
                os.makedirs('result', exist_ok=True)
                pd.DataFrame(test_res).to_csv(f'result/{epoch}-res_test.csv', index=False, header=False)
                pd.DataFrame(test_gts).to_csv(f'result/{epoch}-gt_test.csv', index=False, header=False)
                print('Saved Test CSVs to result/ folder')

            test_met = self.metric_ftns(
                {i: [gt] for i, gt in enumerate(test_gts)},
                {i: [re] for i, re in enumerate(test_res)},
            )
            log.update({('test_' + k): v for k, v in test_met.items()})

        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            val_score = log.get(self.mnt_metric, 0)
            self.lr_scheduler.step(val_score)
        else:
            self.lr_scheduler.step()

        return log