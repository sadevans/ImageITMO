import sys
from datetime import datetime as dtm
import numpy as np
import torch
from logger import logging, logging_images, logging_lr
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


class Trainer:
    def __init__(self, model, cfg, train_options, wandb_run, console):
        self.model = model
        self.cfg = cfg
        self.options = train_options
        self.len_train = len(self.options['train_loader'])
        self.len_val = len(self.options['val_loader'])
        self.lr = cfg.train.optimizer.lr
        self.epoch = 0
        self.step = 0
        self.console = console
        self.logger = wandb_run

        if self.options['rank'] == 0:
            self.logger = wandb_run.get_logger()
            logging(self.logger, metrics='accuracy', value_to_log=0, step=0)
            logging(self.logger, metrics='balanced accuracy', value_to_log=0, step=0)
            logging_lr(self.logger, value_to_log=self.lr, step=0)

    def run_epoch(self, epoch):  # noqa: WPS210, WPS213
        self.model.train()
        self.epoch = epoch
        start_epoch = dtm.now()
        torch.cuda.synchronize()

        all_preds = np.array([])
        all_labels = np.array([])

        run_loss = 0

        # batch iteration
        for step, (batch, targets, _, _) in enumerate(self.options['train_loader']):
            self.step += 1
            start_batch = dtm.now()
            if self.options['mixer']:
                # print(targets)
                if isinstance(targets, list):
                    batch, targets[0] = self.options['mixer'](
                        batch,
                        targets[0].to(torch.long),
                    )
                else:
                    batch, targets = self.options['mixer'](
                        batch,
                        targets.to(torch.long),
                    )

            predicts, labels, loss = self._do_optim_step(batch, targets)
            run_loss += loss.item()
            if isinstance(predicts, list):
                all_preds = np.hstack((all_preds, predicts[0]))
                all_labels = np.hstack((all_labels, labels[0]))
            else:
                all_preds = np.hstack((all_preds, predicts))
                all_labels = np.hstack((all_labels, labels))

            run_acc = sum(all_preds == all_labels) / len(all_preds) * 100.0
            run_balanced_acc = balanced_accuracy_score(all_labels, all_preds) * 100.0

            if self.cfg.train.lr_schedule.name == 'cosine':
                self.options['scheduler'].step()
                self.lr = self.options['scheduler'].get_last_lr()[0]

            delta_batch = dtm.now() - start_batch

            if step == 0:
                if isinstance(labels, list):
                    logging_images(self.logger, batch, labels[0], self.cfg.mapping, per_class=True, is_train=True)
                    logging_images(self.logger, batch, labels[1], self.cfg.mapping1, per_class=True, is_train=True)
                else:
                    logging_images(self.logger, batch, labels, self.cfg.mapping, per_class=True, is_train=True)

            if step % self.cfg.train.freq_vis == 0 and step != 0:
                mean_loss = run_loss / (step + 1)
                if self.options['rank'] == 0:
                    self.console.log(
                        f'Epoch: {self.epoch} / {self.cfg.train.n_epoch}, '
                        + f'batch:[{step} / {self.len_train}], '
                        + f'loss: {mean_loss:.4f}, accuracy: {run_acc:.2f}, '
                        + f'balanced accuracy: {run_balanced_acc:.2f}, lr: {self.lr:.6f}, '
                        + f'time per batch: {delta_batch.total_seconds():.3f} s',
                    )
                    sys.stdout.flush()

                    logging(self.logger, 'loss', mean_loss, step=self.step)
                    logging(self.logger, 'accuracy', run_acc, step=self.step)
                    logging(self.logger, 'balanced accuracy', run_balanced_acc, step=self.step)
                    logging_lr(self.logger, self.lr, step=self.step)

        torch.cuda.synchronize()
        delta_epoch = dtm.now() - start_epoch

        train_acc = sum(all_preds == all_labels) / len(all_preds) * 100.0
        train_balanced_acc = balanced_accuracy_score(all_labels, all_preds) * 100.0
        cm = confusion_matrix(all_labels, all_preds)
        cm_norm = confusion_matrix(all_labels, all_preds, normalize='true')

        if self.cfg.train.lr_schedule.name == 'StepLR':
            self.train_options['scheduler'].step()
            self.lr = self.options['scheduler'].get_last_lr()[0]

        if self.options['rank'] == 0:
            self.console.log(
                f'Mean Loss: {run_loss / self.len_train:.4f}, Final accuracy: {run_acc:.2f}, '
                + f'Final balanced accuracy: {train_balanced_acc:.2f}, lr: {self.lr:.6f}, '
                + f'Total elapsed time: {delta_epoch.total_seconds():.2f} s',
            )
            self.console.print(f'Train process of epoch {epoch} is done', style='bold red')

            logging(self.logger, 'train loss', run_loss / self.len_train, step=self.epoch, per_epoch=True)
            logging(self.logger, 'train accuracy', train_acc, step=self.epoch, per_epoch=True)
            logging(self.logger, 'train balanced accuracy', train_balanced_acc, step=self.epoch, per_epoch=True)
            logging(self.logger, 'train cm', cm, step=self.epoch, per_epoch=True)
            logging(self.logger, 'train norm cm', cm_norm, step=self.epoch, per_epoch=True)

    def eval_epoch(self, epoch):  # noqa: WPS210
        self.model.eval()

        all_preds = np.array([])
        all_labels = np.array([])

        if hasattr(self.cfg.dataset, 'targets_column'):
            if len(self.cfg.dataset.targets_column) > 1:
                all_preds1 = np.array([])
                all_labels1 = np.array([])

        cum_loss = 0
        loss_max = -np.inf
        for i, (batch, targets, _, _) in enumerate(self.options['val_loader']):
            if self.options['use_amp']:
                with torch.cuda.amp.autocast():
                    predicts, labels, loss = self._calculate_loss(batch, targets)
            else:
                predicts, labels, loss = self._calculate_loss(batch, targets)

            cum_loss += loss.item()

            all_preds = np.hstack((all_preds, predicts))
            all_labels = np.hstack((all_labels, labels))

            if i == 0:
                if isinstance(labels, list):
                    logging_images(self.logger, batch, labels[0], self.cfg.mapping, per_class=True, is_train=False)
                    logging_images(self.logger, batch, labels[1], self.cfg.mapping1, per_class=True, is_train=False)
                else:
                    logging_images(self.logger, batch, labels, self.cfg.mapping, per_class=True, is_train=False)

        val_acc = sum(all_preds == all_labels) / len(all_preds) * 100.0
        val_balanced_acc = balanced_accuracy_score(all_labels, all_preds) * 100.0
        cm = confusion_matrix(all_labels, all_preds)
        cm_norm = confusion_matrix(all_labels, all_preds, normalize='true')

        if self.options['rank'] == 0:
            self.console.log(
                f'Val Loss: {cum_loss / self.len_val:.2f}, '
                + f'Val accuracy: {val_acc:.4f}, Val balanced accuracy: {val_balanced_acc:.2f}',
            )

            logging(self.logger, 'val loss', value_to_log=cum_loss / self.len_val, step=self.epoch, is_train=False)
            logging(self.logger, 'val accuracy', value_to_log=val_acc, step=self.epoch, is_train=False)
            logging(self.logger, 'val balanced accuracy', value_to_log=val_balanced_acc, step=self.epoch, is_train=False)
            logging(self.logger, 'val cm', value_to_log=cm, step=self.epoch, is_train=False)
            logging(self.logger, 'val norm cm', value_to_log=cm_norm, step=self.epoch, is_train=False)

            if 'all_preds1' in locals():
                val_acc1 = sum(all_preds1 == all_labels1) / len(all_preds1) * 100.0
                val_balanced_acc1 = balanced_accuracy_score(all_labels1, all_preds1) * 100.0
                cm1 = confusion_matrix(all_labels1, all_preds1)
                cm_norm1 = confusion_matrix(all_labels1, all_preds1, normalize='true')
                self.console.log(
                    f'Val type accuracy: {val_acc1:.4f}, Val type balanced accuracy: {val_balanced_acc1:.2f}',
                )
                logging(self.logger, 'val type accuracy', value_to_log=val_acc1, step=self.epoch, is_train=False)
                logging(self.logger, 'val type balanced accuracy', value_to_log=val_balanced_acc1, step=self.epoch, is_train=False)
                logging(self.logger, 'val type cm', value_to_log=cm1, step=self.epoch, is_train=False)
                logging(self.logger, 'val type norm cm', value_to_log=cm_norm1, step=self.epoch, is_train=False)

            self.console.print(f'Validation of epoch {epoch} is done', style='bold red')
            self.console.print('-------------------------------------------------------', style='bold green')

        return loss

    def _prepare_targets(self, targets):
        if self.cfg.model.num_classes == 1:
            targets = targets.type(torch.cuda.FloatTensor)
            targets = targets.unsqueeze(1)
        if isinstance(targets, list):
            return [target.to(self.options['rank'], non_blocking=True) for target in targets]
        return targets.to(self.options['rank'], non_blocking=True)


    def _calculate_loss(self, batch, targets):
        targets = self._prepare_targets(targets)
        batch = batch.cuda().to(self.options['rank'], memory_format=torch.contiguous_format, non_blocking=True)
        outputs = self.model(batch)

        if self.cfg.train.activation == 'softmax':  # for focal loss
            outputs = torch.softmax(outputs, dim=1)
        loss = self.options['loss'](outputs, targets)

        # we take targets only for main head (1st)
        if isinstance(targets, list) and isinstance(outputs, list):
            if len(targets[0].shape) > 1:  # in case of using mixer
                targets = [torch.max(t, 1)[1] for t in targets]
            targets = [t.cpu().numpy() for t in targets]
            preds = [torch.max(outs, 1)[1] for outs in outputs]
            preds = [p.data.cpu().numpy() for p in preds]
        else:
            if len(targets.shape) > 1:  # in case of using mixer
                _, targets = torch.max(targets, 1)
            targets = targets.cpu().numpy()
            _, preds = torch.max(outputs, 1)
            preds = preds.data.cpu().numpy()

        return preds, targets, loss

    def _do_optim_step(self, batch, targets):
        self.options['optimizer'].zero_grad(set_to_none=True)

        if self.options['use_amp']:
            with torch.cuda.amp.autocast():
                predicts, answers, loss = self._calculate_loss(batch, targets)
            self.options['scaler'].scale(loss).backward()
            self.options['scaler'].step(self.options['optimizer'])
            self.options['scaler'].update()
        else:
            predicts, answers, loss = self._calculate_loss(batch, targets)
            loss.backward()
            self.options['optimizer'].step()

        return predicts, answers, loss
