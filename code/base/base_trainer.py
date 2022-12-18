import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
import torch
import torch.nn as nn
from torch.optim import Adam

from utils import ndcg_k, recall_at_k

class BaseTrainer:
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        submission_dataloader,
        args,
    ):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.submission_dataloader = submission_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(
            self.model.parameters(),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.criterion = nn.BCELoss()

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, mode="valid")

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, mode="test")

    def submission(self, epoch):
        return self.iteration(epoch, self.submission_dataloader, mode="submission")

    def iteration(self, epoch, dataloader, mode="train"):
        raise NotImplementedError

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "RECALL@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "RECALL@10": "{:.4f}".format(recall[1]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
        }
        print(post_fix)

        return [recall[0], ndcg[0], recall[1], ndcg[1]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (
            (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()
        )  # [batch*seq_len]
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

# class BaseTrainer:
#     """
#     Base class for all trainers
#     """
#     def __init__(self, model, criterion, metric_ftns, optimizer, config):
#         self.config = config
#         self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

#         self.model = model
#         self.criterion = criterion
#         self.metric_ftns = metric_ftns
#         self.optimizer = optimizer

#         cfg_trainer = config['trainer']
#         self.epochs = cfg_trainer['epochs']
#         self.save_period = cfg_trainer['save_period']
#         self.monitor = cfg_trainer.get('monitor', 'off')

#         # configuration to monitor model performance and save best
#         if self.monitor == 'off':
#             self.mnt_mode = 'off'
#             self.mnt_best = 0
#         else:
#             self.mnt_mode, self.mnt_metric = self.monitor.split()
#             assert self.mnt_mode in ['min', 'max']

#             self.mnt_best = inf if self.mnt_mode == 'min' else -inf
#             self.early_stop = cfg_trainer.get('early_stop', inf)
#             if self.early_stop <= 0:
#                 self.early_stop = inf

#         self.start_epoch = 1

#         self.checkpoint_dir = config.save_dir

#         # setup visualization writer instance                
#         self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

#         if config.resume is not None:
#             self._resume_checkpoint(config.resume)

#     @abstractmethod
#     def _train_epoch(self, epoch):
#         """
#         Training logic for an epoch

#         :param epoch: Current epoch number
#         """
#         raise NotImplementedError

#     def train(self):
#         """
#         Full training logic
#         """
#         not_improved_count = 0
#         for epoch in range(self.start_epoch, self.epochs + 1):
#             result = self._train_epoch(epoch)

#             # save logged informations into log dict
#             log = {'epoch': epoch}
#             log.update(result)

#             # print logged informations to the screen
#             for key, value in log.items():
#                 self.logger.info('    {:15s}: {}'.format(str(key), value))

#             # evaluate model performance according to configured metric, save best checkpoint as model_best
#             best = False
#             if self.mnt_mode != 'off':
#                 try:
#                     # check whether model performance improved or not, according to specified metric(mnt_metric)
#                     improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
#                                (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
#                 except KeyError:
#                     self.logger.warning("Warning: Metric '{}' is not found. "
#                                         "Model performance monitoring is disabled.".format(self.mnt_metric))
#                     self.mnt_mode = 'off'
#                     improved = False

#                 if improved:
#                     self.mnt_best = log[self.mnt_metric]
#                     not_improved_count = 0
#                     best = True
#                 else:
#                     not_improved_count += 1

#                 if not_improved_count > self.early_stop:
#                     self.logger.info("Validation performance didn\'t improve for {} epochs. "
#                                      "Training stops.".format(self.early_stop))
#                     break

#             if epoch % self.save_period == 0:
#                 self._save_checkpoint(epoch, save_best=best)

#     def _save_checkpoint(self, epoch, save_best=False):
#         """
#         Saving checkpoints

#         :param epoch: current epoch number
#         :param log: logging information of the epoch
#         :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
#         """
#         arch = type(self.model).__name__
#         state = {
#             'arch': arch,
#             'epoch': epoch,
#             'state_dict': self.model.state_dict(),
#             'optimizer': self.optimizer.state_dict(),
#             'monitor_best': self.mnt_best,
#             'config': self.config
#         }
#         filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
#         torch.save(state, filename)
#         self.logger.info("Saving checkpoint: {} ...".format(filename))
#         if save_best:
#             best_path = str(self.checkpoint_dir / 'model_best.pth')
#             torch.save(state, best_path)
#             self.logger.info("Saving current best: model_best.pth ...")

#     def _resume_checkpoint(self, resume_path):
#         """
#         Resume from saved checkpoints

#         :param resume_path: Checkpoint path to be resumed
#         """
#         resume_path = str(resume_path)
#         self.logger.info("Loading checkpoint: {} ...".format(resume_path))
#         checkpoint = torch.load(resume_path)
#         self.start_epoch = checkpoint['epoch'] + 1
#         self.mnt_best = checkpoint['monitor_best']

#         # load architecture params from checkpoint.
#         if checkpoint['config']['arch'] != self.config['arch']:
#             self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
#                                 "checkpoint. This may yield an exception while state_dict is being loaded.")
#         self.model.load_state_dict(checkpoint['state_dict'])

#         # load optimizer state from checkpoint only when optimizer type is not changed.
#         if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
#             self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
#                                 "Optimizer parameters not being resumed.")
#         else:
#             self.optimizer.load_state_dict(checkpoint['optimizer'])

#         self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
