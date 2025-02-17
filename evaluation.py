import torch
import numpy as np
from utils import log_string


class Evaluation:
    """
    Handles evaluation on a given POI dataset and loader.

    The two metrics are MAP and recall@n. Our model predicts sequence of
    next locations determined by the sequence_length at one pass. During evaluation we
    treat each entry of the sequence as single prediction. One such prediction
    is the ranked list of all available locations and we can compute the two metrics.

    As a single prediction is of the size of all available locations,
    evaluation takes its time to compute. The code here is optimized.

    Using the --report_user argument one can access the statistics per user.
    """

    def __init__(self, dataset, dataloader, user_count, h0_strategy, trainer, setting, log):
        self.dataset = dataset
        self.dataloader = dataloader
        self.user_count = user_count
        self.h0_strategy = h0_strategy
        self.trainer = trainer
        self.setting = setting
        self._log = log

    def evaluate(self, suffix):
        self.dataset.reset()
        h = self.h0_strategy.on_init(self.setting.batch_size, self.setting.device)
        if suffix == 'log':
            gjh_log = open("{}_{}.log".format('check_model',suffix), 'w')
        with torch.no_grad():
            iter_cnt = 0
            recall1 = 0
            recall5 = 0
            recall10 = 0
            recall20 = 0
            average_precision = 0.

            u_iter_cnt = np.zeros(self.user_count)
            u_recall1 = np.zeros(self.user_count)
            u_recall5 = np.zeros(self.user_count)
            u_recall10 = np.zeros(self.user_count)
            u_recall20 = np.zeros(self.user_count)
            u_average_precision = np.zeros(self.user_count)
            reset_count = torch.zeros(self.user_count)

            for i, (x, t, t_slot, s, y, y_t, y_t_slot, y_s, reset_h, active_users, t_hour, cat) in enumerate(self.dataloader):
                # if i%50 == 0:
                #     print(i)
                # active_users = active_users.squeeze()
                for j, reset in enumerate(reset_h):
                    if reset:
                        if self.setting.is_lstm:
                            hc = self.h0_strategy.on_reset_test(active_users[0][j], self.setting.device)
                            h[0][0, j] = hc[0]
                            h[1][0, j] = hc[1]
                        else:
                            h[0, j] = self.h0_strategy.on_reset_test(active_users[0][j], self.setting.device)
                        reset_count[active_users[0][j]] += 1

                # squeeze for reasons of "loader-batch-size-is-1"
                x = x.squeeze().to(self.setting.device)
                t = t.squeeze().to(self.setting.device)
                t_slot = t_slot.squeeze().to(self.setting.device)
                t_hour = t_hour.squeeze().to(self.setting.device)
                s = s.squeeze().to(self.setting.device)
                cat = cat.squeeze().to(self.setting.device)

                y = y.squeeze()
                y_t = y_t.squeeze().to(self.setting.device)
                y_t_slot = y_t_slot.squeeze().to(self.setting.device)
                y_s = y_s.squeeze().to(self.setting.device)
                active_users = active_users.to(self.setting.device)
                negative_pair_g = self.dataloader.dataset.negative_pair_g.to(self.setting.device)
                positive_pair_g = self.dataloader.dataset.positive_pair_g.to(self.setting.device)
                g = self.dataloader.dataset.block.to(self.setting.device)
                hour_loc_g = self.dataloader.dataset.hour_loc_g.to(self.setting.device)
                node_index_key = self.dataloader.dataset.node_index_key

                # evaluate:
                out, h = self.trainer.evaluate(x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_users,g, positive_pair_g, negative_pair_g,t_hour,node_index_key,cat)
                time = t
                for j in range(self.setting.batch_size):
                    # o contains a per user list of votes for all locations for each sequence entry
                    o = out[j]

                    # partition elements
                    o_n = o.cpu().detach().numpy()
                    ind = np.argpartition(o_n, -20, axis=1)[:, -10:]  # top 10 elements

                    y_j = y[:, j]
                    x_j = x[:, j]
                    time_j = time[:, j]

                    for k in range(len(y_j)):
                        if reset_count[active_users[0][j]] > 1:
                            continue  # skip already evaluated users.

                        # resort indices for k:
                        ind_k = ind[k]
                        r = ind_k[np.argsort(-o_n[k, ind_k], axis=0)]  # sort top 10 elements descending
                        top10_scores = o_n[k, r]
                        r = torch.tensor(r)
                        t = y_j[k]

                        if suffix == 'log':
                            log_s = "{},{},{},{},{},{}".format(active_users[0][j].item(),time_j[k].item(),x_j[k].item(),t.item(),",".join(map(str, r.tolist())) , ",".join(map(str, top10_scores)) )
                            gjh_log.write(log_s + '\n')
                            gjh_log.flush()
                            

                        # compute MAP:
                        r_kj = o_n[k, :]
                        t_val = r_kj[t]
                        upper = np.where(r_kj > t_val)[0]
                        precision = 1. / (1 + len(upper))

                        # store
                        u_iter_cnt[active_users[0][j]] += 1
                        u_recall1[active_users[0][j]] += t in r[:1]
                        u_recall5[active_users[0][j]] += t in r[:5]
                        u_recall10[active_users[0][j]] += t in r[:10]
                        # u_recall20[active_users[0][j]] += t in r[:20]
                        u_average_precision[active_users[0][j]] += precision

                # self.trainer.update_memory(g)
            
            # if suffix == 'log':
                
            formatter = "{0:.8f}"
            for j in range(self.user_count):
                iter_cnt += u_iter_cnt[j]
                recall1 += u_recall1[j]
                recall5 += u_recall5[j]
                recall10 += u_recall10[j]
                # recall20 += u_recall20[j]
                average_precision += u_average_precision[j]

                if self.setting.report_user > 0 and (j + 1) % self.setting.report_user == 0:
                    print('Report user', j, 'preds:', u_iter_cnt[j], 'recall@1',
                          formatter.format(u_recall1[j] / u_iter_cnt[j]), 'MAP',
                          formatter.format(u_average_precision[j] / u_iter_cnt[j]), sep='\t')

            # print('recall@1:', formatter.format(recall1 / iter_cnt))
            # print('recall@5:', formatter.format(recall5 / iter_cnt))
            # print('recall@10:', formatter.format(recall10 / iter_cnt))
            # print('MAP', formatter.format(average_precision / iter_cnt))
            # print('predictions:', iter_cnt)

            log_string(self._log, 'recall@1: ' + formatter.format(recall1 / iter_cnt))
            log_string(self._log, 'recall@5: ' + formatter.format(recall5 / iter_cnt))
            log_string(self._log, 'recall@10: ' + formatter.format(recall10 / iter_cnt))
            # log_string(self._log, 'recall@20: ' + formatter.format(recall20 / iter_cnt))
            log_string(self._log, 'MAP: ' + formatter.format(average_precision / iter_cnt))
            print('predictions:', iter_cnt)
