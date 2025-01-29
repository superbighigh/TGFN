import torch
import torch.nn as nn
from enum import Enum
import time
import numpy as np
from utils import *
import scipy.sparse as sp
import math
from modules import MemoryModule, MemoryOperation, MsgLinkPredictor, TemporalTransformerConv, TimeEncode, Time2Vec
import dgl
import copy



class Rnn(Enum):
    """ The available RNN units """

    RNN = 0
    GRU = 1
    LSTM = 2

    @staticmethod
    def from_string(name):
        if name == 'rnn':
            return Rnn.RNN
        if name == 'gru':
            return Rnn.GRU
        if name == 'lstm':
            return Rnn.LSTM
        raise ValueError('{} not supported in --rnn'.format(name))


class RnnFactory():
    """ Creates the desired RNN unit. """

    def __init__(self, rnn_type_str):
        self.rnn_type = Rnn.from_string(rnn_type_str)

    def __str__(self):
        if self.rnn_type == Rnn.RNN:
            return 'Use pytorch RNN implementation.'
        if self.rnn_type == Rnn.GRU:
            return 'Use pytorch GRU implementation.'
        if self.rnn_type == Rnn.LSTM:
            return 'Use pytorch LSTM implementation.'

    def is_lstm(self):
        return self.rnn_type in [Rnn.LSTM]

    def create(self, hidden_size):
        if self.rnn_type == Rnn.RNN:
            return nn.RNN(hidden_size, hidden_size)
        if self.rnn_type == Rnn.GRU:
            return nn.GRU(hidden_size, hidden_size)
        if self.rnn_type == Rnn.LSTM:
            return nn.LSTM(hidden_size, hidden_size)


class Flashback(nn.Module):
    """ Flashback RNN: Applies weighted average using spatial and tempoarl data in combination
    of user embeddings to the output of a generic RNN unit (RNN, GRU, LSTM).
    """

    def __init__(self, input_size, user_count, hidden_size, f_t, f_s, f_h, f_diy, rnn_factory, lambda_loc, lambda_user, use_weight, graph, spatial_graph, friend_graph, use_graph_user, use_spatial_graph, interact_graph, sq_count, sp_count, cat_count,setting,cat_g,cat_relation_graph):
        super().__init__()
        self.input_size = input_size  # POI个数
        self.user_count = user_count
        self.cat_count = cat_count
        self.hidden_size = hidden_size
        self.f_t = f_t  # function for computing temporal weight
        self.f_s = f_s  # function for computing spatial weight
        self.f_h = f_h
        self.f_diy = f_diy

        self.lambda_loc = lambda_loc
        self.lambda_user = lambda_user
        self.use_weight = use_weight
        self.use_graph_user = use_graph_user

        self.memory_dim = hidden_size
        self.edge_feat_dim = hidden_size
        self.temporal_dim = hidden_size
        self.embedding_dim = hidden_size
        self.num_heads = 10
        self.n_neighbors = 10
        self.memory_updater_type = 'gru'
        self.num_nodes = input_size + user_count
        self.layers = 1
        self.cat_emb_count = 1
        self.setting = setting

        self.time2v = Time2Vec('sin', hidden_size)
        self.temporal_encoder = TimeEncode(self.temporal_dim)

        self.memory = MemoryModule(self.num_nodes,self.memory_dim)

        self.embedding_attn = TemporalTransformerConv(self.edge_feat_dim,
                                                      self.memory_dim,
                                                      self.temporal_encoder,
                                                      self.embedding_dim,
                                                      self.num_heads,
                                                      setting,
                                                      layers=self.layers,
                                                      allow_zero_in_degree=True,
                                                      sq_count=sq_count,
                                                      sp_count=sp_count)

        self.memory_ops = MemoryOperation(self.memory_updater_type,self.memory,self.edge_feat_dim,self.temporal_encoder,self.embedding_attn)

        self.msg_linkpredictor = MsgLinkPredictor(self.embedding_dim)
        self.hour_encoder = nn.Embedding(12, hidden_size)
        self.encoder = nn.Embedding(input_size, hidden_size)  # location embedding
        # weight = torch.FloatTensor(input_size, hidden_size)
        # nn.init.xavier_uniform_(weight)
        # self.encoder.weight = nn.Parameter(weight)
        # normalize_emb = F.normalize(self.encoder.weight.data, p=2, dim=1)
        # self.encoder.weight.data = normalize_emb
        # self.time_encoder = nn.Embedding(24 * 7, hidden_size)  # time embedding
        self.user_encoder = nn.Embedding(user_count, hidden_size)  # user embedding
        # user_weight = torch.FloatTensor(input_size, hidden_size)
        # nn.init.xavier_uniform_(user_weight)
        # self.user_encoder.weight = nn.Parameter(user_weight)
        # user_normalize_emb = F.normalize(self.user_encoder.weight.data, p=2, dim=1)
        # self.user_encoder.weight.data = user_normalize_emb
        self.category_graph = sparse_matrix_to_tensor(calculate_random_walk_matrix(
                cat_g))
        self.cat_relation_graph = cat_relation_graph
        self.cat_encoder = nn.Embedding(cat_count, hidden_size)  # category embedding
        # pre_train_model = torch.load('pre_model_25.ckpt', map_location="cpu")
        # print(pre_train_model['model']['encoder.weight'].shape)
        # self.encoder = nn.Embedding.from_pretrained(pre_train_model['model']['encoder.weight'])
        # self.encoder.requires_grad_(False)
        # self.encoder.cuda()
        # print(self.encoder(3))
        # self.user_encoder = nn.Embedding.from_pretrained(pre_train_model['model']['user_encoder.weight'])
        # self.user_encoder.requires_grad_(False)
        # print(torch.cat([pre_train_model['model']['encoder.weight'],pre_train_model['model']['user_encoder.weight']]).shape)
        # self.encoder = nn.Embedding.from_pretrained(torch.cat([pre_train_model['model']['encoder.weight'],pre_train_model['model']['user_encoder.weight']]))
        # self.user_encoder.cuda()
        # self.encoder = nn.Embedding(input_size+user_count, hidden_size)
        # self.user_encoder = nn.Embedding(user_count, hidden_size)
        # ent_weight = torch.FloatTensor(input_size+user_count, hidden_size)
        # nn.init.xavier_uniform(ent_weight)
        # self.encoder.weight = nn.Parameter(ent_weight)
        # normalize_ent_emb = F.normalize(self.encoder.weight.data, p=2, dim=1)
        # self.encoder.weight.data = normalize_ent_emb
        self.rnn = rnn_factory.create(hidden_size*self.cat_emb_count)
        # self.fc_temp = nn.Linear((self.cat_emb_count+2) * hidden_size, (self.cat_emb_count+1) * hidden_size)
        self.fc = nn.Linear((self.cat_emb_count+1) * hidden_size, input_size)
        # self.fc = nn.Linear(hidden_size, input_size)

        # # device = torch.device("cuda")
        # device = torch.device("cuda")
        # self.I = identity(graph.shape[0], format='coo')
        # self.graph = sparse_matrix_to_tensor(
        #     calculate_random_walk_matrix((graph * self.lambda_loc + self.I).astype(np.float32)))
        # self.interact_graph = sparse_matrix_to_tensor(calculate_random_walk_matrix(
        #         interact_graph))  # (M, N)
        # loc_emb = self.encoder(torch.LongTensor(list(range(self.input_size))).to(device))
        # self.encoder_weight = torch.sparse.mm(self.graph.to(device), loc_emb).to(device)  # (input_size, hidden_size)

        # interact_graph = self.interact_graph.to(device)
        # self.encoder_weight_user = torch.sparse.mm(
        #     interact_graph, loc_emb).to(device)


    def forward(self, x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_user, g, positive_graph, negative_pair_g, model_type,t_hour, node_index_key,cat):
        
        seq_len, user_len = x.size()
        if self.setting.gcn :
            memory = self.memory.memory.to(x.device)

            nid = g.ndata[dgl.NID].to(x.device)
            start_time = time.time()
            user_nid = nid[nid < self.user_count]
            user_em = self.user_encoder(user_nid).to(x.device)
            loc_nid = nid[nid>= self.user_count]
            loc_em = self.encoder(loc_nid-self.user_count).to(x.device)
            all2em = dict(zip(torch.cat([user_nid,loc_nid],dim=0).tolist(),torch.arange(len(user_nid)+len(loc_nid))))

            base_embedding_temp = torch.cat([user_em,loc_em],dim=0)
            end_time = time.time()
            # print("batch start",1, end_time-start_time)
            node_id = [all2em[int(n)] for n in g.ndata[dgl.NID]]
            base_embedding = base_embedding_temp[node_id]


            # hour_g_loc_nid = hour_loc_g.ndata[dgl.NID].to(x.device)
            # hour_g_loc_em = self.encoder(hour_g_loc_nid).to(x.device)
            hour_g_loc_em = base_embedding

            cat_emb = self.cat_encoder(torch.LongTensor(
                list(range(self.cat_count))).to(x.device))
            emb_memory = memory[nid, :]
            emb_t = g.ndata['timestamp']
            embedding, memory_hid, cat_emb = self.embedding_attn(g, emb_memory, emb_t, base_embedding, self.cat_relation_graph.to(x.device), cat_emb)
            
            if self.setting.cat_emb_type == 'transition':
                category_graph = self.category_graph.to(x.device)
                cat_encoder_weight = torch.sparse.mm(category_graph, cat_emb).to(x.device)
                new_cat_emb = []
                for i in range(seq_len):
                    # (user_len, hidden_size)
                    temp_cat = torch.index_select(cat_encoder_weight, 0, cat[i])
                    new_cat_emb.append(temp_cat)
                cat_emb = torch.stack(new_cat_emb, dim=0)
            elif self.setting.cat_emb_type == 'relation':
                new_cat_emb = []
                for i in range(seq_len):
                    # (user_len, hidden_size)
                    temp_cat = torch.index_select(cat_emb, 0, cat[i])
                    new_cat_emb.append(temp_cat)
                cat_emb = torch.stack(new_cat_emb, dim=0)
            else:
                print("not allow cat_emb_type:",self.setting.cat_emb_type)
                exit()
            
            emb2pred = dict(
                zip(g.ndata['node_index'].tolist(), g.nodes().tolist()))
            
            new_x_emb = []
            new_x_emb_hour = []
            for i in range(seq_len):
                # (user_len, hidden_size)
                x_id = []
                for n in range(user_len):
                    if node_index_key[int(x[i][n])+self.user_count] < (n*seq_len+i):
                        x_id.append(emb2pred[int((n*seq_len+i)*(self.num_nodes)+int(x[i][n]))+self.user_count])
                        # print(int((n*seq_len+i)*(self.num_nodes)+int(x[i][n]))+self.user_count)
                    else:
                        x_id.append(emb2pred[int(x[i][n])+self.user_count])

                
                temp_x = embedding[x_id]
                new_x_emb.append(temp_x)
            x_emb = torch.stack(new_x_emb, dim=0).to(x.device)
            category_graph = self.category_graph.to(x.device)
            cat_emb = self.cat_encoder(torch.LongTensor(
                list(range(self.cat_count))).to(x.device))
            cat_encoder_weight = torch.sparse.mm(category_graph, cat_emb).to(
                x.device)  # (input_size, hidden_size)
            new_cat_emb = []
            for i in range(seq_len):
                # (user_len, hidden_size)
                temp_cat = torch.index_select(cat_encoder_weight, 0, cat[i])
                new_cat_emb.append(temp_cat)

            cat_emb = torch.stack(new_cat_emb, dim=0)
            u_id = [emb2pred[int(n)] for n in active_user[0]]
            user_preference = embedding[u_id]
            # # p_u = self.user_encoder(active_user)  # (1, user_len, hidden_size)
            # user_preference = user_preference.view(user_len, self.hidden_size).to(x.device)
        else:
            x_emb = self.encoder(x)

        p_u = self.encoder(active_user)  # (1, user_len, hidden_size)
        # print(p_u)
        p_u = p_u.view(user_len, self.hidden_size)
        # print(user_preference.size())
        # user_loc_similarity = torch.exp(
        #     -(torch.norm(user_preference - x_emb, p=2, dim=-1))).to(x.device)
        # user_loc_similarity = user_loc_similarity.permute(1, 0)

        out, h = self.rnn(x_emb, h)  # (seq_len, user_len, hidden_size)
        out_w = torch.zeros(seq_len, user_len,
                            self.hidden_size*self.cat_emb_count, device=x.device)

        for i in range(seq_len):
            sum_w = torch.zeros(user_len, 1, device=x.device)  # (200, 1)
            for j in range(i + 1):
                a_j = 1.0 
                c_j = 1.0
                if self.setting.dist_t == "time2v":
                    time_encode_i = self.time2v(t[i].unsqueeze(dim=1).float())
                    time_encode_j = self.time2v(t[j].unsqueeze(dim=1).float())
                    a_j = torch.exp(-torch.sum(torch.abs(time_encode_i - time_encode_j) ** 2, 1))
                    # a_j = self.f_diy(dist_em,user_len)
                elif self.setting.dist_t == "temporal":
                    time_encode_i = self.temporal_encoder(t[i].unsqueeze(dim=1).double()).view(t[i].shape[0], -1)
                    time_encode_j = self.temporal_encoder(t[j].unsqueeze(dim=1).double()).view(t[j].shape[0], -1)
                    a_j = torch.exp(-torch.sum(torch.abs(time_encode_i - time_encode_j) ** 2, 1))
                elif self.setting.dist_t == "cat":
                    dist_t = t[i] - t[j]
                    a_j = self.f_t(dist_t, user_len)
                    c_j = torch.exp(-torch.sum(torch.abs(cat_emb[i] - cat_emb[j]) ** 2, 1))
                    c_j = c_j.unsqueeze(1)
                elif self.setting.dist_t == "cat+time2v":
                    time_encode_i = self.time2v(t[i].unsqueeze(dim=1).float())
                    time_encode_j = self.time2v(t[j].unsqueeze(dim=1).float())
                    a_j = torch.exp(-torch.sum(torch.abs(time_encode_i - time_encode_j) ** 2, 1))
                    c_j = torch.exp(-torch.sum(torch.abs(cat_emb[i] - cat_emb[j]) ** 2, 1))
                    c_j = c_j.unsqueeze(1)
                else:
                    dist_t = t[i] - t[j]
                    a_j = self.f_t(dist_t, user_len)  # (user_len, )
                # a_j = self.f_h(dist_t,dist_h, user_len)
                dist_s = torch.norm(s[i] - s[j], dim=-1)
                b_j = self.f_s(dist_s, user_len)
                a_j = a_j.unsqueeze(1)  # (user_len, 1)
                b_j = b_j.unsqueeze(1)
                if self.setting.wj_type == 'mul':
                    w_j = a_j * b_j * c_j + 1e-10  # small epsilon to avoid 0 division
                elif self.setting.wj_type == 'half-add':
                    w_j = a_j * b_j + c_j + 1e-10
                elif self.setting.wj_type == 'add':
                    w_j = a_j + b_j + c_j + 1e-10
                # w_j = w_j * user_loc_similarity[:, j].unsqueeze(1)  # (user_len, 1)
                sum_w += w_j
                out_w[i] += w_j * out[j]  # (user_len, hidden_size)
            out_w[i] /= sum_w

        out_pu = torch.zeros(seq_len, user_len, (self.cat_emb_count+1) *
                             self.hidden_size, device=x.device)
        if self.setting.test_model != 'no':
            out_pu = out_pu * 0
        for i in range(seq_len):
            # (user_len, hidden_size * 2)
            # print(out_w[i].shape)
            # print(p_u.shape)
            # time_encode_i = self.time2v(t[i].unsqueeze(dim=1).float())
            out_pu[i] = torch.cat([out_w[i], p_u], dim=1)

        # y_linear_temp = self.fc_temp(out_pu)
        y_linear = self.fc(out_pu)  # (seq_len, user_len, loc_count)
        # y_linear = self.fc(out_w)
        return y_linear, h
        # 正负样本
        # return y_linear, h, pred_pos,pred_neg

    def update_memory(self, subg):
        base_embedding = self.encoder(subg.ndata[dgl.NID])
        new_g = self.memory_ops(subg, base_embedding)
        # nid = new_g.ndata[dgl.NID].to(self.memory.memory.device)
        # memory = new_g.ndata['memory'].to(self.memory.memory.device)
        # self.memory.set_memory(nid, memory)
        # self.memory.set_last_update_t(
        #     new_g.ndata[dgl.NID].to(self.memory.last_update_t.device), new_g.ndata['timestamp'].to(self.memory.last_update_t.device))

    # Some memory operation wrappers
    def detach_memory(self):
        self.memory.detach_memory()

    def reset_memory(self):
        self.memory.reset_memory()

    def store_memory(self):
        memory_checkpoint = {}
        memory_checkpoint['memory'] = copy.deepcopy(self.memory.memory)
        memory_checkpoint['last_t'] = copy.deepcopy(self.memory.last_update_t)
        return memory_checkpoint

    def restore_memory(self, memory_checkpoint):
        self.memory.memory = memory_checkpoint['memory']
        self.memory.last_update_time = memory_checkpoint['last_t']

'''
~~~ h_0 strategies ~~~
Initialize RNNs hidden states
'''


def create_h0_strategy(hidden_size, is_lstm):
    if is_lstm:
        return LstmStrategy(hidden_size, FixNoiseStrategy(hidden_size), FixNoiseStrategy(hidden_size))
    else:
        return FixNoiseStrategy(hidden_size)


class H0Strategy():

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def on_init(self, user_len, device):
        pass

    def on_reset(self, user):
        pass

    def on_reset_test(self, user, device):
        return self.on_reset(user)


class FixNoiseStrategy(H0Strategy):
    """ use fixed normal noise as initialization """

    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        mu = 0
        sd = 1 / self.hidden_size
        self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu

    def on_init(self, user_len, device):
        hs = []
        for i in range(user_len):
            hs.append(self.h0)
        # (1, 200, 10)
        return torch.stack(hs, dim=0).view(1, user_len, self.hidden_size).to(device)

    def on_reset(self, user):
        return self.h0


class LstmStrategy(H0Strategy):
    """ creates h0 and c0 using the inner strategy """

    def __init__(self, hidden_size, h_strategy, c_strategy):
        super(LstmStrategy, self).__init__(hidden_size)
        self.h_strategy = h_strategy
        self.c_strategy = c_strategy

    def on_init(self, user_len, device):
        h = self.h_strategy.on_init(user_len, device)
        c = self.c_strategy.on_init(user_len, device)
        return h, c

    def on_reset(self, user):
        h = self.h_strategy.on_reset(user)
        c = self.c_strategy.on_reset(user)
        return h, c
