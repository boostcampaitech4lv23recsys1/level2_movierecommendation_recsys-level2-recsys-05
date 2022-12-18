import numpy as np
import tqdm
from base import BaseTrainer

class PretrainTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        submission_dataloader,
        args,
    ):
        super(PretrainTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
        )

    def pretrain(self, epoch, pretrain_dataloader):

        desc = (
            f"AAP-{self.args.aap_weight}-"
            f"MIP-{self.args.mip_weight}-"
            f"MAP-{self.args.map_weight}-"
            f"SP-{self.args.sp_weight}"
        )

        pretrain_data_iter = tqdm.tqdm(
            enumerate(pretrain_dataloader),
            desc=f"{self.args.model_name}-{self.args.data_name} Epoch:{epoch}",
            total=len(pretrain_dataloader),
            bar_format="{l_bar}{r_bar}",
        )

        self.model.train()
        aap_loss_avg = 0.0
        mip_loss_avg = 0.0
        map_loss_avg = 0.0
        sp_loss_avg = 0.0

        for i, batch in pretrain_data_iter:
            # 0. batch_data will be sent into the device(GPU or CPU)
            batch = tuple(t.to(self.device) for t in batch)
            (
                attributes,
                masked_item_sequence,
                pos_items,
                neg_items,
                masked_segment_sequence,
                pos_segment,
                neg_segment,
            ) = batch

            aap_loss, mip_loss, map_loss, sp_loss = self.model.pretrain(
                attributes,
                masked_item_sequence,
                pos_items,
                neg_items,
                masked_segment_sequence,
                pos_segment,
                neg_segment,
            )

            joint_loss = (
                self.args.aap_weight * aap_loss
                + self.args.mip_weight * mip_loss
                + self.args.map_weight * map_loss
                + self.args.sp_weight * sp_loss
            )

            self.optim.zero_grad()
            joint_loss.backward()
            self.optim.step()

            aap_loss_avg += aap_loss.item()
            mip_loss_avg += mip_loss.item()
            map_loss_avg += map_loss.item()
            sp_loss_avg += sp_loss.item()

        num = len(pretrain_data_iter) * self.args.pre_batch_size
        losses = {
            "epoch": epoch,
            "aap_loss_avg": aap_loss_avg / num,
            "mip_loss_avg": mip_loss_avg / num,
            "map_loss_avg": map_loss_avg / num,
            "sp_loss_avg": sp_loss_avg / num,
        }
        print(desc)
        print(str(losses))
        return losses


class FinetuneTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        submission_dataloader,
        args,
    ):
        super(FinetuneTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
        )

    def iteration(self, epoch, dataloader, mode="train"):

        # Setting the tqdm progress bar

        rec_data_iter = tqdm.tqdm(
            enumerate(dataloader),
            desc="Recommendation EP_%s:%d" % (mode, epoch),
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}",
        )
        if mode == "train":
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, target_pos, target_neg, _ = batch
                # Binary cross_entropy
                sequence_output = self.model.finetune(input_ids)
                loss = self.cross_entropy(sequence_output, target_pos, target_neg)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": "{:.4f}".format(rec_cur_loss),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

        else:
            self.model.eval()

            pred_list = None
            answer_list = None
            for i, batch in rec_data_iter:

                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, _, target_neg, answers = batch
                recommend_output = self.model.finetune(input_ids)

                recommend_output = recommend_output[:, -1, :]

                rating_pred = self.predict_full(recommend_output)

                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                ind = np.argpartition(rating_pred, -10)[:, -10:]

                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

                batch_pred_list = ind[
                    np.arange(len(rating_pred))[:, None], arr_ind_argsort
                ]

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(
                        answer_list, answers.cpu().data.numpy(), axis=0
                    )

            if mode == "submission":
                return pred_list
            else:
                return self.get_full_sort_score(epoch, answer_list, pred_list)


# class Trainer(BaseTrainer):
#     """
#     Trainer class
#     """
#     def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
#                  data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
#         super().__init__(model, criterion, metric_ftns, optimizer, config)
#         self.config = config
#         self.device = device
#         self.data_loader = data_loader
#         if len_epoch is None:
#             # epoch-based training
#             self.len_epoch = len(self.data_loader)
#         else:
#             # iteration-based training
#             self.data_loader = inf_loop(data_loader)
#             self.len_epoch = len_epoch
#         self.valid_data_loader = valid_data_loader
#         self.do_validation = self.valid_data_loader is not None
#         self.lr_scheduler = lr_scheduler
#         self.log_step = int(np.sqrt(data_loader.batch_size))

#         self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
#         self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

#     def _train_epoch(self, epoch):
#         """
#         Training logic for an epoch

#         :param epoch: Integer, current training epoch.
#         :return: A log that contains average loss and metric in this epoch.
#         """
#         self.model.train()
#         self.train_metrics.reset()
#         for batch_idx, (data, target) in enumerate(self.data_loader):
#             data, target = data.to(self.device), target.to(self.device)

#             self.optimizer.zero_grad()
#             output = self.model(data)
#             loss = self.criterion(output, target)
#             loss.backward()
#             self.optimizer.step()

#             self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
#             self.train_metrics.update('loss', loss.item())
#             for met in self.metric_ftns:
#                 self.train_metrics.update(met.__name__, met(output, target))

#             if batch_idx % self.log_step == 0:
#                 self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
#                     epoch,
#                     self._progress(batch_idx),
#                     loss.item()))
#                 self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

#             if batch_idx == self.len_epoch:
#                 break
#         log = self.train_metrics.result()

#         if self.do_validation:
#             val_log = self._valid_epoch(epoch)
#             log.update(**{'val_'+k : v for k, v in val_log.items()})

#         if self.lr_scheduler is not None:
#             self.lr_scheduler.step()
#         return log

#     def _valid_epoch(self, epoch):
#         """
#         Validate after training an epoch

#         :param epoch: Integer, current training epoch.
#         :return: A log that contains information about validation
#         """
#         self.model.eval()
#         self.valid_metrics.reset()
#         with torch.no_grad():
#             for batch_idx, (data, target) in enumerate(self.valid_data_loader):
#                 data, target = data.to(self.device), target.to(self.device)

#                 output = self.model(data)
#                 loss = self.criterion(output, target)

#                 self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
#                 self.valid_metrics.update('loss', loss.item())
#                 for met in self.metric_ftns:
#                     self.valid_metrics.update(met.__name__, met(output, target))
#                 self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

#         # add histogram of model parameters to the tensorboard
#         for name, p in self.model.named_parameters():
#             self.writer.add_histogram(name, p, bins='auto')
#         return self.valid_metrics.result()

#     def _progress(self, batch_idx):
#         base = '[{}/{} ({:.0f}%)]'
#         if hasattr(self.data_loader, 'n_samples'):
#             current = batch_idx * self.data_loader.batch_size
#             total = self.data_loader.n_samples
#         else:
#             current = batch_idx
#             total = self.len_epoch
#         return base.format(current, total, 100.0 * current / total)