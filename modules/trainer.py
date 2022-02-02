import logging
import os
from abc import abstractmethod

import torch
from numpy import inf
from .utils import con_loss as contrastive_loss



class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)

  
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.epochs = self.args.epochs
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.n_gpu = args.n_gpu
        self.checkpoint_dir = args.save_dir

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        best_epoch = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    cur_metric = log['val_BLEU_4'] + 0.33 * log['val_BLEU_1'] + 0.67 * log['val_METEOR']
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and cur_metric >= self.mnt_best)
                               #(self.mnt_mode == 'max' and (log[self.mnt_metric]) >= self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = cur_metric
                    #self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                    best_epoch = epoch
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
            print('best performance in epoch: ',best_epoch)

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        self.logger.info('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

        self.logger.info('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader,
                 val_dataloader, test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.img_con_loss_weight = args.weight_img_con_loss
        self.txt_con_loss_weight = args.weight_txt_con_loss
        self.img_bce_loss_weight = args.weight_img_bce_loss
        self.txt_bce_loss_weight = args.weight_txt_bce_loss

    def _train_epoch(self, epoch):

        self.logger.info('[{}/{}] Start to train in the training set.'.format(epoch, self.epochs))
        ce_loss = 0
        img_con_loss = 0
        txt_con_loss = 0
        img_bce_loss = 0
        txt_bce_loss = 0
        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks, labels) in enumerate(self.train_dataloader):

            images, reports_ids, reports_masks, labels = images.to(self.device), reports_ids.to(self.device), \
                                                 reports_masks.to(self.device), labels.to(self.device)

            output, img_con_ls, txt_con_ls, img_cls, txt_cls = self.model(images, reports_ids, labels=labels, mode='train')
            img_bce_ls = self.bce_loss(img_cls, labels)
            txt_bce_ls = self.bce_loss(txt_cls, labels)
            ce_ls = self.criterion(output, reports_ids, reports_masks)
            #print('222', con_ls, con_ls.shape)
            #if self.n_gpu > 1:
            img_con_ls = img_con_ls.mean()
            txt_con_ls = txt_con_ls.mean()
            #con_ls = contrastive_loss(memory_matrix, labels)
            #con_ls = 0
            img_con_loss += img_con_ls.item()
            txt_con_loss += txt_con_ls.item()
            img_bce_loss += img_bce_ls.item()
            txt_bce_loss += txt_bce_ls.item()
            #con_loss += 0
            ce_loss += ce_ls.item()
            loss = ce_ls + self.img_con_loss_weight * img_con_ls + self.txt_con_loss_weight * txt_con_ls\
                   + self.img_bce_loss_weight * img_bce_ls + self.txt_bce_loss_weight * txt_bce_ls
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.args.log_period == 0:
                self.logger.info('[{}/{}] Step: {}/{}, CE Ls: {:.5f}, CON Ls1: {:.5f}, CON Ls2: {:.5f}, BCE Ls1: {:.5f}, BCE Ls2: {:.5f}.'
                                 .format(epoch, self.epochs, batch_idx, len(self.train_dataloader),
                                         ce_loss / (batch_idx + 1), img_con_loss / (batch_idx + 1),
                                         txt_con_loss / (batch_idx + 1),
                                         img_bce_loss / (batch_idx + 1),
                                         txt_bce_ls / (batch_idx + 1)))

        log = {'ce_loss': ce_loss / len(self.train_dataloader), 'img_con': img_con_loss / len(self.train_dataloader),
               'txt_con': txt_con_loss / len(self.train_dataloader),
               'img_bce_loss': img_bce_loss / len(self.train_dataloader), 'txt_bce_loss': txt_bce_loss / len(self.train_dataloader)}

        self.logger.info('[{}/{}] Start to evaluate in the validation set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, labels) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks, labels = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device), labels.to(self.device)

                output, _ = self.model(images, labels = labels, mode='sample')
                # change to self.model.module for multi-gpu
                if self.n_gpu>1:
                    reports = self.model.module.tokenizer.decode_batch(output.cpu().numpy())
                    ground_truths = self.model.module.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                else:
                    reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                    ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)

            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.logger.info('[{}/{}] Start to evaluate in the test set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, labels) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks, labels = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device), labels.to(self.device)
                output, _ = self.model(images, labels=labels, mode='sample')
                if self.n_gpu>1:
                    reports = self.model.module.tokenizer.decode_batch(output.cpu().numpy())
                    ground_truths = self.model.module.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                else:
                    reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                    ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        self.lr_scheduler.step()

        return log
