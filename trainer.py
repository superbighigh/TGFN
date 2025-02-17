import time

import torch
import torch.nn as nn
import numpy as np
from utils import *
from network import Flashback
from scipy.sparse import csr_matrix


class FlashbackTrainer():
    """ Instantiates Flashback module with spatial and temporal weight functions.
    Performs loss computation and prediction.
    """

    def __init__(self, lambda_t, lambda_s, lambda_loc, lambda_user, use_weight, transition_graph, spatial_graph,
                 friend_graph, use_graph_user, use_spatial_graph, interact_graph, sq_count, sp_count, setting, cat_g, cat_relation_graph):
        """ The hyper parameters to control spatial and temporal decay.
        """
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s

        self.lambda_loc = lambda_loc
        self.lambda_user = lambda_user
        self.use_weight = use_weight
        self.use_graph_user = use_graph_user
        self.use_spatial_graph = use_spatial_graph
        self.graph = transition_graph
        self.spatial_graph = spatial_graph
        self.friend_graph = friend_graph
        self.interact_graph = interact_graph
        self.sq_count = sq_count
        self.sp_count = sp_count
        self.setting = setting
        self.cat_g = cat_g
        self.cat_relation_graph = cat_relation_graph

    def __str__(self):
        return 'Use flashback training.'

    def count_parameters(self):
        param_count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_count += param.numel()
        return param_count
    
    def parameters(self):
        return self.model.parameters()

    def prepare(self, loc_count, user_count, cat_count, hidden_size, gru_factory, device):
        def f_t(delta_t, user_len): return ((torch.cos(delta_t * 2 * np.pi / 86400) + 1) / 2) * torch.exp(
            -(delta_t / 86400 * self.lambda_t))  # hover cosine + exp decay

        def f_h(delta_t, delta_h, user_len): return ((torch.cos(delta_h * 2 * np.pi / 12) + 1) / 2) * torch.exp(
            -(delta_t / 86400 * self.lambda_t))  # hover cosine + exp decay
        def f_diy(delta_t, delta_em, user_len): return (delta_em * torch.exp(
            -(delta_t / 86400 * self.lambda_t)))  # hover cosine + exp decay
        # exp decay  2个functions
        def f_s(delta_s, user_len): return torch.exp(-(delta_s * self.lambda_s))
        self.loc_count = loc_count
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        # self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.criterion = torch.nn.BCELoss(reduction='mean')
        self.model = Flashback(loc_count, user_count, hidden_size, f_t, f_s, f_h, f_diy, gru_factory, self.lambda_loc,self.lambda_user, self.use_weight, self.graph, self.spatial_graph, self.friend_graph,self.use_graph_user, self.use_spatial_graph, self.interact_graph,self.sq_count,self.sp_count,cat_count,self.setting,self.cat_g, self.cat_relation_graph).to(device)

    def evaluate(self, x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_users, g, positive_pair_g, negative_pair_g,t_hour, node_index_key, cat):
        """ takes a batch (users x location sequence)
        then does the prediction and returns a list of user x sequence x location
        describing the probabilities for each location at each position in the sequence.
        t, s are temporal and spatial data related to the location sequence x
        y_t, y_s are temporal and spatial data related to the target sequence y.
        Flashback does not access y_t and y_s for prediction!
        """

        self.model.eval()
        # (seq_len, user_len, loc_count)
        # out, h, pred_pos,pred_neg = self.model(x, t, t_slot, s, y_t,
        #                     y_t_slot, y_s, h, active_users, g, positive_pair_g, negative_pair_g, 'test')
        out, h = self.model(x, t, t_slot, s, y_t,
                            y_t_slot, y_s, h, active_users, g, positive_pair_g, negative_pair_g, 'test', t_hour, node_index_key, cat)

        out_t = out.transpose(0, 1)
        return out_t, h  # model outputs logits

    def loss(self, x, t, t_slot, s, y, y_t, y_t_slot, y_s, h, active_users, g, positive_pair_g, negative_pair_g, t_hour, node_index_key, cat):
        """ takes a batch (users x location sequence)
        and corresponding targets in order to compute the training loss """

        self.model.train()

        out, h = self.model(x, t, t_slot, s, y_t, y_t_slot, y_s, h,
                            active_users, g, positive_pair_g, negative_pair_g, 'train', t_hour, node_index_key,cat)  # out (seq_len, batch_size, loc_count)
        # out, h, pred_pos, pred_neg = self.model(x, t, t_slot, s, y_t, y_t_slot, y_s, h,
        #                     active_users, g, positive_pair_g, negative_pair_g, 'train')
        out = out.view(-1, self.loc_count)  # (seq_len * batch_size, loc_count)

        y = y.view(-1)  # (seq_len * batch_size)
        # print("get loss1", time.time())
        l = self.cross_entropy_loss(out, y)
        # print("get loss2", time.time())
        # loss = self.criterion(pred_pos, torch.ones_like(pred_pos))
        # loss += self.criterion(pred_neg, torch.zeros_like(pred_neg))
        # print(l,loss)
        return l

    def update_memory(self, subg):
        self.model.update_memory(subg)

    # Some memory operation wrappers
    def detach_memory(self):
        self.model.detach_memory()

    def reset_memory(self):
        self.model.reset_memory()

    def store_memory(self):
        return self.model.store_memory()

    def restore_memory(self, memory_checkpoint):
        self.model.restore_memory(memory_checkpoint)