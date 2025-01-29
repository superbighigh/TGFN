import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.base import DGLError
from dgl.ops import edge_softmax
import dgl.function as fn


class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive.
    (Identity has already been supported by PyTorch 1.2, we will directly
    import torch.nn.Identity in the future)
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x


# class MsgLinkPredictor(nn.Module):
#     """Predict Pair wise link from pos subg and neg subg
#     use message passing.

#     Use Two layer MLP on edge to predict the link probability

#     Parameters
#     ----------
#     embed_dim : int
#         dimension of each each feature's embedding

#     Example
#     ----------
#     >>> linkpred = MsgLinkPredictor(10)
#     >>> pos_g = dgl.graph(([0,1,2,3,4],[1,2,3,4,0]))
#     >>> neg_g = dgl.graph(([0,1,2,3,4],[2,1,4,3,0]))
#     >>> x = torch.ones(5,10)
#     >>> linkpred(x,pos_g,neg_g)
#     (tensor([[0.0902],
#          [0.0902],
#          [0.0902],
#          [0.0902],
#          [0.0902]], grad_fn=<AddmmBackward>),
#     tensor([[0.0902],
#          [0.0902],
#          [0.0902],
#          [0.0902],
#          [0.0902]], grad_fn=<AddmmBackward>))
#     """

#     def __init__(self, emb_dim):
#         super(MsgLinkPredictor, self).__init__()
#         self.src_fc = nn.Linear(emb_dim, emb_dim)
#         self.dst_fc = nn.Linear(emb_dim, emb_dim)
#         self.out_fc = nn.Linear(emb_dim, 1)

#     def link_pred(self, edges):
#         src_hid = self.src_fc(edges.src['embedding'])
#         dst_hid = self.dst_fc(edges.dst['embedding'])
#         score = F.relu(src_hid+dst_hid)
#         score = self.out_fc(score)
#         return {'score': score}

#     def forward(self, x, pos_g, neg_g):
#         # Local Scope?
#         pos_g.ndata['embedding'] = x
#         neg_g.ndata['embedding'] = x

#         pos_g.apply_edges(self.link_pred)
#         neg_g.apply_edges(self.link_pred)

#         pos_escore = pos_g.edata['score']
#         neg_escore = neg_g.edata['score']
#         return pos_escore, neg_escore
#         # return pos_escore

class MsgLinkPredictor(nn.Module):

    def __init__(self, emb_dim):
        super(MsgLinkPredictor, self).__init__()
        self.src_fc = nn.Linear(emb_dim, emb_dim)
        self.dst_fc = nn.Linear(emb_dim, emb_dim)
        self.out_fc = nn.Linear(emb_dim, 1)

    def link_pred(self, edges):
        # src_hid = self.src_fc(edges.src['embedding'])
        # dst_hid = self.dst_fc(edges.dst['embedding'])
        score = torch.exp(-(torch.norm(edges.src['embedding'] - edges.dst['embedding'], p=2, dim=-1)))
        return {'score': score}

    def forward(self, x, pos_g, neg_g):
        # Local Scope?
        pos_g.ndata['embedding'] = x
        neg_g.ndata['embedding'] = x

        pos_g.apply_edges(self.link_pred)
        neg_g.apply_edges(self.link_pred)

        pos_escore = pos_g.edata['score']
        neg_escore = neg_g.edata['score']
        return pos_escore, neg_escore
        # return pos_escore



class TimeEncode(nn.Module):
    """Use finite fourier series with different phase and frequency to encode
    time different between two event

    ..math::
        \Phi(t) = [\cos(\omega_0t+\psi_0),\cos(\omega_1t+\psi_1),...,\cos(\omega_nt+\psi_n)] 

    Parameter
    ----------
    dimension : int
        Length of the fourier series. The longer it is , 
        the more timescale information it can capture

    Example
    ----------
    >>> tecd = TimeEncode(10)
    >>> t = torch.tensor([[1]])
    >>> tecd(t)
    tensor([[[0.5403, 0.9950, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 1.0000]]], dtype=torch.float64, grad_fn=<CosBackward>)
    """

    def __init__(self, dimension):
        super(TimeEncode, self).__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)
        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                           .double().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).double())

    def forward(self, t):
        t = t.unsqueeze(dim=2)
        output = torch.cos(self.w(t)).float()
        return output


class MemoryModule(nn.Module):
    """Memory module as well as update interface

    The memory module stores both historical representation in last_update_t

    Parameters
    ----------
    n_node : int
        number of node of the entire graph

    hidden_dim : int
        dimension of memory of each node

    Example
    ----------
    Please refers to examples/pytorch/tgn/tgn.py;
                     examples/pytorch/tgn/train.py 

    """

    def __init__(self, n_node, hidden_dim):
        super(MemoryModule, self).__init__()
        self.n_node = n_node
        self.hidden_dim = hidden_dim
        self.reset_memory()

    def reset_memory(self):
        self.last_update_t = nn.Parameter(torch.zeros(
            self.n_node).float(), requires_grad=False)
        self.memory = nn.Parameter(torch.zeros(
            (self.n_node, self.hidden_dim)).float(), requires_grad=False)
        # self.memory = torch.randn((self.n_node, self.hidden_dim), requires_grad=False)

    def backup_memory(self):
        """
        Return a deep copy of memory state and last_update_t
        For test new node, since new node need to use memory upto validation set
        After validation, memory need to be backed up before run test set without new node
        so finally, we can use backup memory to update the new node test set
        """
        return self.memory.clone(), self.last_update_t.clone()

    def restore_memory(self, memory_backup):
        """Restore the memory from validation set

        Parameters
        ----------
        memory_backup : (memory,last_update_t)
            restore memory based on input tuple
        """
        self.memory = memory_backup[0].clone()
        self.last_update_t = memory_backup[1].clone()

    # Which is used for attach to subgraph
    def get_memory(self, node_idxs):
        return self.memory[node_idxs, :]

    # When the memory need to be updated
    def set_memory(self, node_idxs, values):
        try:
            self.memory[node_idxs, :] = values
        except Exception as e:
            print(node_idxs.device, values.device, self.memory.device)
            raise e
        

    def set_last_update_t(self, node_idxs, values):
        self.last_update_t[node_idxs] = values

    # For safety check
    def get_last_update(self, node_idxs):
        return self.last_update_t[node_idxs]

    def detach_memory(self):
        """
        Disconnect the memory from computation graph to prevent gradient be propagated multiple
        times
        """
        self.memory.detach_()


class MemoryOperation(nn.Module):
    """ Memory update using message passing manner, update memory based on positive
    pair graph of each batch with recurrent module GRU or RNN

    Message function
    ..math::
        m_i(t) = concat(memory_i(t^-),TimeEncode(t),v_i(t))

    v_i is node feature at current time stamp

    Aggregation function
    ..math::
        \bar{m}_i(t) = last(m_i(t_1),...,m_i(t_b))

    Update function
    ..math::
        memory_i(t) = GRU(\bar{m}_i(t),memory_i(t-1))

    Parameters
    ----------

    updater_type : str
        indicator string to specify updater

        'rnn' : use Vanilla RNN as updater

        'gru' : use GRU as updater

    memory : MemoryModule
        memory content for update

    e_feat_dim : int
        dimension of edge feature

    temporal_dim : int
        length of fourier series for time encoding

    Example
    ----------
    Please refers to examples/pytorch/tgn/tgn.py
    """

    def __init__(self, updater_type, memory, e_feat_dim, temporal_encoder, embedding_attn):
        super(MemoryOperation, self).__init__()
        updater_dict = {'gru': nn.GRUCell, 'rnn': nn.RNNCell}
        self.memory = memory
        memory_dim = self.memory.hidden_dim
        self.temporal_encoder = temporal_encoder
        self.message_dim = memory_dim+memory_dim + \
            e_feat_dim+self.temporal_encoder.dimension
        self.updater = updater_dict[updater_type](input_size=self.message_dim,
                                                  hidden_size=memory_dim)
        self.memory = memory
        self.embedding_attn = embedding_attn
        self.res_fc = Identity()

    # Here assume g is a subgraph from each iteration
    def stick_feat_to_graph(self, g):
        # How can I ensure order of the node ID
        g.ndata['timestamp'] = self.memory.last_update_t[g.ndata[dgl.NID].to('cpu')].to(g.device)
        g.ndata['memory'] = self.memory.memory[g.ndata[dgl.NID].to('cpu')].to(g.device)

    def msg_fn_cat(self, edges):
        src_delta_time = edges.data['timestamp'] - edges.src['timestamp']
        time_encode = self.temporal_encoder(src_delta_time.unsqueeze(
            dim=1)).view(len(edges.data['timestamp']), -1)
        ret = torch.cat([edges.src['memory'], edges.dst['memory'],
                         edges.data['feats'], time_encode], dim=1)
        return {'message': ret, 'timestamp': edges.data['timestamp']}

    def agg_last(self, nodes):
        timestamp, latest_idx = torch.max(nodes.mailbox['timestamp'], dim=1)
        ret = nodes.mailbox['message'].gather(1, latest_idx.repeat(
            self.message_dim).view(-1, 1, self.message_dim)).view(-1, self.message_dim)
        return {'message_bar': ret.reshape(-1, self.message_dim), 'timestamp': timestamp}

    def update_memory(self, nodes):
        # It should pass the feature through RNN
        ret = self.updater(
            nodes.data['message_bar'].float(), nodes.data['memory'].float())
        return {'memory': ret}

    def forward(self, g, base_embedding):
        # self.stick_feat_to_graph(g)
        # g.update_all(self.msg_fn_cat, self.agg_last, self.update_memory)
        memory = self.memory.memory.to(g.device)
        nid = g.ndata[dgl.NID].to(g.device)
        emb_memory = memory[nid, :]
        emb_t = g.ndata['timestamp']
        embedding, memory_hid = self.embedding_attn(g, emb_memory, emb_t, base_embedding.to(g.device))
        # embedding_memory = self.res_fc(embedding)
        emb2pred = dict(
            zip(g.ndata[dgl.NID].tolist(), g.nodes().tolist()))
        # print(positive_graph.ndata[dgl.NID])
        feat_id = [emb2pred[int(n)] for n in g.ndata[dgl.NID]]
        feat = memory_hid[feat_id]
        # print(g.ndata[dgl.NID])
        # print(feat[0])
        self.memory.set_memory(nid.to(self.memory.memory.device), feat.to(self.memory.memory.device))
        self.memory.set_last_update_t(nid.to(self.memory.last_update_t.device), emb_t.float().to(self.memory.last_update_t.device))
        # print(self.memory.memory[9331])
        return g


class EdgeGATConv(nn.Module):
    '''Edge Graph attention compute the graph attention from node and edge feature then aggregate both node and
    edge feature.

    Parameter
    ==========
    node_feats : int
        number of node features

    edge_feats : int
        number of edge features

    out_feats : int
        number of output features

    num_heads : int
        number of heads in multihead attention

    feat_drop : float, optional
        drop out rate on the feature

    attn_drop : float, optional
        drop out rate on the attention weight

    negative_slope : float, optional
        LeakyReLU angle of negative slope.

    residual : bool, optional
        whether use residual connection

    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.

    '''

    def __init__(self,
                 node_feats,
                 edge_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False):
        super(EdgeGATConv, self).__init__()
        self._num_heads = num_heads
        self._node_feats = node_feats
        self._edge_feats = edge_feats
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.fc_node = nn.Linear(
            self._node_feats, self._out_feats*self._num_heads)
        self.fc_edge = nn.Linear(
            self._edge_feats, self._out_feats*self._num_heads)
        self.attn_l = nn.Parameter(torch.FloatTensor(
            size=(1, self._num_heads, self._out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(
            size=(1, self._num_heads, self._out_feats)))
        self.attn_e = nn.Parameter(torch.FloatTensor(
            size=(1, self._num_heads, self._out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.residual = residual
        if residual:
            if self._node_feats != self._out_feats:
                self.res_fc = nn.Linear(
                    self._node_feats, self._out_feats*self._num_heads, bias=False)
            else:
                self.res_fc = Identity()
        self.reset_parameters()
        self.activation = activation
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_node.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if self.residual and isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def msg_fn(self, edges):
        ret = edges.data['a'].view(-1, self._num_heads,
                                   1)*edges.data['el_prime']
        return {'m': ret}

    def forward(self, graph, nfeat, efeat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            nfeat = self.feat_drop(nfeat)
            efeat = self.feat_drop(efeat)

            node_feat = self.fc_node(
                nfeat).view(-1, self._num_heads, self._out_feats)
            edge_feat = self.fc_edge(
                efeat).view(-1, self._num_heads, self._out_feats)

            el = (node_feat*self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (node_feat*self.attn_r).sum(dim=-1).unsqueeze(-1)
            ee = (edge_feat*self.attn_e).sum(dim=-1).unsqueeze(-1)
            graph.ndata['ft'] = node_feat
            graph.ndata['el'] = el
            graph.ndata['er'] = er
            graph.edata['ee'] = ee
            # print(ee)
            graph.apply_edges(fn.u_add_e('el', 'ee', 'el_prime'))
            graph.apply_edges(fn.e_add_v('el_prime', 'er', 'e'))
            e = self.leaky_relu(graph.edata['e'])
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # print(graph.edata['a'])
            graph.edata['efeat'] = edge_feat
            graph.update_all(self.msg_fn, fn.sum('m', 'ft'))
            rst = graph.ndata['ft']
            # print(rst[0])
            if self.residual:
                # resval = self.res_fc(nfeat).view(
                #     nfeat.shape[0], -1, self._out_feats)
                resval = self.res_fc(nfeat).view(
                    nfeat.shape[0], self._out_feats, -1)
                # print(rst[0])
                # print(resval[0])
                
                rst = (rst + resval).view(nfeat.shape[0], self._out_feats)
                # print(rst[0])
                # exit()

            if self.activation:
                rst = self.activation(rst)
            # print(rst.size())
            # exit()
            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst

class EdgeGATConvDiy(nn.Module):

    def __init__(self,
                 node_feats,
                 edge_feats,
                 out_feats,
                 num_heads,
                 setting,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 sq_count=50,
                 sp_count=50,
                 score_type='time-transh'
                 ):
        super(EdgeGATConvDiy, self).__init__()
        self._num_heads = num_heads
        self._node_feats = node_feats
        self._edge_feats = edge_feats
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.sq_count = sq_count
        self.sp_count = sp_count
        self.score_type = score_type
        self.setting = setting
        

        rel_weight = torch.FloatTensor(7, out_feats)
        norm_weight = torch.FloatTensor(7, out_feats)
        proj_weight = torch.FloatTensor(7, out_feats * out_feats)
        nn.init.xavier_uniform_(rel_weight)
        nn.init.xavier_uniform_(norm_weight)
        nn.init.xavier_uniform_(proj_weight)
        self.rel_embeddings = nn.Embedding(7, out_feats)
        self.norm_embeddings = nn.Embedding(7, out_feats)
        self.proj_embeddings = nn.Embedding(7, out_feats * out_feats)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)
        self.norm_embeddings.weight = nn.Parameter(norm_weight)
        self.proj_embeddings.weight = nn.Parameter(proj_weight)
        normalize_rel_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
        normalize_norm_emb = F.normalize(self.norm_embeddings.weight.data, p=2, dim=1)
        self.rel_embeddings.weight.data = normalize_rel_emb
        self.norm_embeddings.weight.data = normalize_norm_emb

        self.weekday_embeddings = nn.Embedding(8, out_feats)
        self.month_embeddings = nn.Embedding(13, out_feats)
        self.hour_embeddings = nn.Embedding(25, out_feats)
        weekday_weight = torch.FloatTensor(8, out_feats)
        month_weight = torch.FloatTensor(13, out_feats)
        hour_weight = torch.FloatTensor(25, out_feats)
        nn.init.xavier_uniform_(weekday_weight)
        nn.init.xavier_uniform_(month_weight)
        nn.init.xavier_uniform_(hour_weight)
        self.weekday_embeddings.weight = nn.Parameter(weekday_weight)
        self.month_embeddings.weight = nn.Parameter(month_weight)
        self.hour_embeddings.weight = nn.Parameter(hour_weight)
        normalize_weekday_emb = F.normalize(self.weekday_embeddings.weight.data, p=2, dim=1)
        normalize_month_emb = F.normalize(self.month_embeddings.weight.data, p=2, dim=1)
        normalize_hour_emb = F.normalize(self.hour_embeddings.weight.data, p=2, dim=1)
        self.weekday_embeddings.weight.data = normalize_weekday_emb
        self.month_embeddings.weight.data = normalize_month_emb
        self.hour_embeddings.weight.data = normalize_hour_emb

        self.time_norm = nn.Linear(out_feats,out_feats)
        self.time_rel = nn.Linear(out_feats,out_feats)
        self.time_emb = nn.Linear(2*out_feats,out_feats)
        self.relu = nn.ReLU()
        nn.init.xavier_uniform_(self.time_rel.weight)
        if self.time_rel.bias is not None:
            nn.init.zeros_(self.time_rel.bias)

    def projection_transH_pytorch(self, original, norm):
        return original - torch.sum(original * norm, dim=len(original.size())-1, keepdim=True) * norm
    
    def projection_transR_pytorch(self, original, proj_matrix):
        ent_embedding_size = original.shape[1]
        rel_embedding_size = proj_matrix.shape[1] // ent_embedding_size
        original = original.view(-1, ent_embedding_size, 1)
        proj_matrix = proj_matrix.view(-1, rel_embedding_size, ent_embedding_size)
        return torch.matmul(proj_matrix, original).view(-1, rel_embedding_size)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_node.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if self.residual and isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def msg_score_no(self, edges):  
        fin_score = torch.from_numpy(np.ones(len(edges),dtype=np.float32 )).to(edges.src['nfeat'].device)
        return {'score': fin_score.float()}

    def msg_score_transr(self, edges):
        proj_h_e = self.projection_transR_pytorch(edges.src['nfeat'], edges.data['proj'])
        proj_t_e = self.projection_transR_pytorch(edges.dst['nfeat'], edges.data['proj'])
        rel = edges.data['rel']
        score = torch.sum((proj_h_e + rel - proj_t_e) ** 2, 1)
        return {'score': score} 

    def msg_score_transe(self, edges):
        proj_h_e = edges.src['nfeat']
        proj_t_e = edges.dst['nfeat']
        rel = edges.data['rel']
        score = torch.exp(-torch.sum(torch.abs(proj_h_e + rel - proj_t_e) ** 2, 1))
        fin_score = score
        return {'score': fin_score}

    def msg_score_transh(self, edges):
        proj_h_e = self.projection_transH_pytorch(edges.src['nfeat'], edges.data['norm'])
        proj_t_e = self.projection_transH_pytorch(edges.dst['nfeat'], edges.data['norm'])
        rel = edges.data['rel']
        score = torch.exp(-torch.sum(torch.abs(proj_h_e + rel - proj_t_e) ** 2, 1))

        fin_score = score
        return {'score': fin_score}

    def msg_score(self, edges):
        proj_h_e = self.projection_transH_pytorch(edges.src['nfeat'], edges.data['norm'])
        proj_t_e = self.projection_transH_pytorch(edges.dst['nfeat'], edges.data['norm'])
        rel = edges.data['rel']
        score = torch.exp(-torch.sum(torch.abs(proj_h_e + rel - proj_t_e) ** 2, 1))

        # # proj_h_e_t = self.projection_transH_pytorch(edges.src['nfeat'], edges.src['norm'])
        # proj_t_e_t = self.projection_transH_pytorch(edges.dst['time_emb'], edges.data['norm'])
        # # rel_t = edges.src['rel']
        # score_t = torch.exp(-torch.sum(torch.abs(proj_h_e + rel - proj_t_e_t) ** 2, 1))

        proj_h_e_t = self.projection_transH_pytorch(edges.src['nfeat'], edges.data['time_norm'])
        proj_t_e_t = self.projection_transH_pytorch(edges.dst['nfeat'], edges.data['time_norm'])
        rel_t = edges.data['time_rel']
        score_t = torch.exp(-torch.sum(torch.abs(proj_h_e_t + rel_t - proj_t_e_t) ** 2, 1))

        # proj_h_weekday_e = self.projection_transH_pytorch(edges.src['nfeat'], edges.data['weekday_norm'])
        # proj_t_weekday_e = self.projection_transH_pytorch(edges.data['weekday_feat'], edges.data['weekday_norm'])
        # weekday_rel = edges.data['weekday_rel']
        # weekday_score = torch.sum(torch.abs(proj_h_weekday_e + weekday_rel - proj_t_weekday_e) ** 2, 1)

        # proj_h_month_e = self.projection_transH_pytorch(edges.src['nfeat'], edges.data['month_norm'])
        # proj_t_month_e = self.projection_transH_pytorch(edges.data['month_feat'], edges.data['month_norm'])
        # month_rel = edges.data['month_rel']
        # month_score = torch.sum(torch.abs(proj_h_month_e + month_rel - proj_t_month_e) ** 2, 1)
        # fin_score = score+score_t
        fin_score = score*score_t
        # fin_score = score
        
        # fin_score = score * weekday_score * month_score
        # print(rel.size())
        # return {'score': score,'weekday_score': weekday_score,'month_score':month_score}
        return {'score': fin_score}

    def mesg_score_hour(self, edges):
        hour_em = self.hour_embeddings(edges.src[dgl.NID])
        proj_h_e = self.projection_transH_pytorch(hour_em, edges.data['hour_norm'])
        proj_t_e = self.projection_transH_pytorch(edges.dst['nfeat'], edges.data['hour_norm'])
        rel = edges.data['hour_rel']
        score = torch.exp(-torch.sum(torch.abs(proj_h_e + rel - proj_t_e) ** 2, 1)) * edges.data['num']
        return {'score': score}

    def msg_fn(self, edges):
        # print(edges.src['nfeat'].size(), edges.data['a'].size())
        ret = edges.src['nfeat']*edges.data['a'].view(-1,1)

        return {'m': ret}

    def emb_mix(self, time_type, time_id):
        if time_type == 'month':
            time_period = 12
            emb_layer = self.month_embeddings
        elif time_type == 'week':
            time_period = 7
            emb_layer = self.weekday_embeddings
        elif time_type == 'hour':
            time_period = 6
            emb_layer = self.hour_embeddings
        last_time_id = (time_id-1+time_period)%time_period
        next_time_id = (time_id+1)%time_period
        last_time_emb = emb_layer(last_time_id).to(last_time_id.device)
        time_emb = emb_layer(time_id).to(time_id.device)
        next_time_emb = emb_layer(next_time_id).to(next_time_id.device)
        mix_emb = (last_time_emb+time_emb+next_time_emb)/3
        return mix_emb


    def forward(self, graph, nfeat, efeat, cat_relation_graph, cat_embedding, get_attention=False):
        with graph.local_scope(), cat_relation_graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            

            # temp_emb = torch.cat([nfeat,graph.ndata['time_emb']],dim=1)
            
            graph.ndata['nfeat'] = nfeat
            # graph.ndata['nfeat'] = self.relu(self.time_emb(temp_emb))
            # graph.ndata['nfeat'] = temp_emb
            # print(graph.edata['feats_type'].tolist())
            # print(graph.ndata['nfeat'])
            # exit()
            graph.edata['rel'] = self.rel_embeddings(graph.edata['feats_type']).to(nfeat.device)
            graph.edata['norm'] = self.norm_embeddings(graph.edata['feats_type']).to(nfeat.device)
            graph.edata['proj'] = self.proj_embeddings(graph.edata['feats_type']).to(nfeat.device)
            # graph.ndata['time_norm'] = self.relu(self.time_norm(graph.ndata['time_emb']))
            time_type_list = torch.from_numpy(np.ones(graph.num_edges(),dtype=np.int64 )*5).to(nfeat.device)
            graph.edata['time_norm'] = self.norm_embeddings(time_type_list).to(nfeat.device)
            
            if self.setting.cat_emb_type == "transition":
                cat_hid = cat_embedding
            elif self.setting.cat_emb_type == "relation":
                cat_relation_graph.ndata['nfeat'] = cat_embedding
                cat_type_list = torch.from_numpy(np.ones(cat_relation_graph.num_edges(),dtype=np.int64 )*6).to(nfeat.device)
                # cat_relation_graph.edata['norm'] = self.norm_embeddings(cat_type_list).to(nfeat.device)
                cat_relation_graph.edata['rel'] = self.rel_embeddings(cat_type_list).to(nfeat.device)
                if self.setting.cat_score_type == "transe":
                    cat_relation_graph.apply_edges(self.msg_score_transe)
                elif self.setting.cat_score_type == 'no':
                    cat_relation_graph.apply_edges(self.msg_score_no)
                cat_relation_graph.edata['a'] = edge_softmax(cat_relation_graph, cat_relation_graph.edata['score'])
                cat_relation_graph.update_all(self.msg_fn, fn.sum('m', 'nfeat'))
                cat_hid = cat_relation_graph.ndata['nfeat']
            else:
                print("not allow cat_emb_type:",self.setting.cat_emb_type)
                exit()

            
            # graph.edata['time_rel'] = F.normalize(self.time_rel(graph.edata['time_emb']), p=2, dim=1)
            # graph.edata['time_rel'] = graph.edata['time_emb']

            if self.score_type == 'time-transh':
                graph.edata['time_rel'] = F.normalize(self.relu(self.time_rel(graph.edata['time_emb'])), p=2, dim=1)
                graph.apply_edges(self.msg_score)
            elif self.score_type == 'transh':
                graph.apply_edges(self.msg_score_transh)
            elif self.score_type == 'no':
                graph.apply_edges(self.msg_score_no)
            elif self.score_type == 'transe':
                graph.apply_edges(self.msg_score_transe)
            elif self.score_type == 'transr':
                graph.apply_edges(self.msg_score_transr)
            elif self.score_type == 'time-transh-slot':
                graph.edata['weekday_rel'] = self.weekday_embeddings(graph.edata['weekday']).to(nfeat.device)
                graph.edata['month_rel'] = self.month_embeddings(graph.edata['month']).to(nfeat.device)
                graph.edata['hour_rel'] = self.hour_embeddings(graph.edata['hour']).to(nfeat.device)
                graph.edata['time_rel'] = (graph.edata['weekday_rel'] + graph.edata['month_rel'] + graph.edata['hour_rel'])/3
                graph.apply_edges(self.msg_score)
            elif self.score_type == 'time-transh-slot-mix':
                graph.edata['weekday_rel'] = self.emb_mix('week',graph.edata['weekday'].to(nfeat.device))
                graph.edata['month_rel'] = self.emb_mix('month',graph.edata['month'].to(nfeat.device))
                graph.edata['hour_rel'] = self.emb_mix('hour',graph.edata['hour'].to(nfeat.device))
                graph.edata['time_rel'] = (graph.edata['weekday_rel'] + graph.edata['month_rel'] + graph.edata['hour_rel'])/3
                graph.apply_edges(self.msg_score)
            elif self.score_type == 'time-transh-slot-month':
                graph.edata['time_rel'] = self.month_embeddings(graph.edata['month']).to(nfeat.device)
                graph.apply_edges(self.msg_score)
            elif self.score_type == 'time-transh-slot-day':
                graph.edata['time_rel'] = self.weekday_embeddings(graph.edata['weekday']).to(nfeat.device)
                graph.apply_edges(self.msg_score)
            elif self.score_type == 'time-transh-slot-hour':
                graph.edata['time_rel'] = self.hour_embeddings(graph.edata['hour']).to(nfeat.device)
                graph.apply_edges(self.msg_score)
            graph.edata['a'] = edge_softmax(graph, graph.edata['score'])

            # # 开始优化
            # score = graph.edata['a']  # (num_edges,)
            # dst = graph.edges()[1]        # (num_edges,)
            # rel = graph.edata['feats_type']  # (num_edges,)
            # num_edges = score.shape[0]
            # device = score.device

            # # 计算 group_id
            # num_relations = torch.max(rel).item() + 1
            # group_id = dst * num_relations + rel

            # # 根据 feats_type 设置每条边的保留数量 k_per_group
            # k_per_group = {}
            # if self.sq_count > -1:
            #     k_per_group[1] = self.sq_count
            # if self.sp_count > -1:
            #     k_per_group[2] = self.sp_count
            # # 对于其他 feats_type，可以设置默认值，假设为保留所有边
            # default_k = num_edges  # 一个较大的数，确保保留所有边

            # # 对 group_id 进行排序，获取排序后的索引
            # group_id_sorted, sort_indices = group_id.sort()
            # score_sorted = score[sort_indices]
            # rel_sorted = rel[sort_indices]
            # group_id_sorted = group_id_sorted.to(device)

            # # 找到每个 group 的边界
            # group_boundaries = torch.cat((
            #     torch.tensor([0], device=device),
            #     (group_id_sorted[1:] != group_id_sorted[:-1]).nonzero(as_tuple=False).squeeze() + 1,
            #     torch.tensor([num_edges], device=device)
            # ))

            # # 初始化一个布尔掩码
            # keep_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)

            # # 对每个 group 进行处理
            # for i in range(len(group_boundaries) - 1):
            #     start = group_boundaries[i].item()
            #     end = group_boundaries[i + 1].item()
            #     group_scores = score_sorted[start:end]
            #     group_rels = rel_sorted[start:end]
            #     group_size = end - start

            #     # 获取当前组的 feats_type
            #     group_rel = group_rels[0].item()
            #     k = k_per_group.get(group_rel, default_k)
            #     k = min(k, group_size)  # 确保 k 不超过组内边的数量

            #     # 对组内的边按 score 降序排序，获取前 k 个的索引
            #     topk_indices = group_scores.topk(k, largest=True, sorted=False).indices + start

            #     # 更新掩码
            #     keep_mask[topk_indices] = True

            # # 恢复到原始顺序
            # keep_mask_original = torch.zeros(num_edges, dtype=torch.bool, device=device)
            # keep_mask_original[sort_indices] = keep_mask

            # # print(graph.edata['score'].tolist())
            # # 更新边的权重
            # graph.edata['a'] = torch.where(keep_mask_original, score, torch.tensor(0.0, device=device))

            # # print(graph.edata['score'].tolist())
            # # exit()
            # # 继续后续操作

            
            # print(graph.ndata['nfeat'].tolist())
            # graph.update_all(self.msg_fn, fn.sum('m', 'nfeat'))
            # rst = graph.ndata['nfeat']

            # print(rst.tolist())
            # exit()

            # graph.edata['a'] = edge_softmax(graph, graph.edata['score'])
            graph.update_all(self.msg_fn, fn.sum('m', 'nfeat'))
            rst = graph.ndata['nfeat']

            
            
            # return rst, hour_graph.ndata['nfeat']
            return rst,cat_hid

class TemporalEdgePreprocess(nn.Module):
    '''Preprocess layer, which finish time encoding and concatenate 
    the time encoding to edge feature.

    Parameter
    ==========
    edge_feats : int
        number of orginal edge feature

    temporal_encoder : torch.nn.Module
        time encoder model
    '''

    def __init__(self, edge_feats, temporal_encoder):
        super(TemporalEdgePreprocess, self).__init__()
        self.edge_feats = edge_feats
        self.temporal_encoder = temporal_encoder

    def edge_fn(self, edges):
        t0 = torch.zeros_like(edges.dst['timestamp'])
        time_diff = edges.data['timestamp'] - edges.src['timestamp']
        time_encode = self.temporal_encoder(
            time_diff.unsqueeze(dim=1)).view(t0.shape[0], -1)
        edge_feat = torch.cat([edges.data['feats'], time_encode], dim=1)
        return {'efeat': edge_feat}

    def forward(self, graph):
        graph.apply_edges(self.edge_fn)
        efeat = graph.edata['efeat']
        return efeat


class TemporalTransformerConv(nn.Module):
    def __init__(self,
                 edge_feats,
                 memory_feats,
                 temporal_encoder,
                 out_feats,
                 num_heads,
                 setting,
                 allow_zero_in_degree=False,
                 layers=1,
                 sq_count=50,
                 sp_count=50):
        '''Temporal Transformer model for TGN and TGAT

        Parameter
        ==========
        edge_feats : int
            number of edge features

        memory_feats : int
            dimension of memory vector

        temporal_encoder : torch.nn.Module
            compute fourier time encoding

        out_feats : int
            number of out features

        num_heads : int
            number of attention head

        allow_zero_in_degree : bool, optional
            If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
            since no message will be passed to those nodes. This is harmful for some applications
            causing silent performance regression. This module will raise a DGLError if it detects
            0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
            and let the users handle it by themselves. Defaults: ``False``.
        '''
        super(TemporalTransformerConv, self).__init__()
        self._edge_feats = edge_feats
        self._memory_feats = memory_feats
        self.temporal_encoder = temporal_encoder
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads
        self.layers = layers

        self.preprocessor = TemporalEdgePreprocess(
            self._edge_feats, self.temporal_encoder)
        self.layer_list = nn.ModuleList()
        self.layer_list.append(EdgeGATConvDiy(node_feats=self._memory_feats,
                                           edge_feats=self._edge_feats+self.temporal_encoder.dimension,
                                           out_feats=self._out_feats,
                                           num_heads=self._num_heads,
                                           setting=setting,
                                           feat_drop=0.,
                                           attn_drop=0.,
                                           residual=True,
                                           allow_zero_in_degree=allow_zero_in_degree,
                                           sq_count=sq_count,
                                           sp_count=sp_count,
                                           score_type=setting.score_type,
                                           ))
        for i in range(self.layers-1):
            self.layer_list.append(EdgeGATConv(node_feats=self._out_feats*self._num_heads,
                                               edge_feats=self._edge_feats+self.temporal_encoder.dimension,
                                               out_feats=self._out_feats,
                                               num_heads=self._num_heads,
                                               feat_drop=0.6,
                                               attn_drop=0.6,
                                               residual=True,
                                               allow_zero_in_degree=allow_zero_in_degree))

        self.base_fc = nn.Linear(memory_feats, memory_feats)
        self.memory_fc = nn.Linear(memory_feats, memory_feats)
        self.emb_fc = nn.Linear(memory_feats, memory_feats)
        self.memory_decode_fc = nn.Linear(memory_feats, memory_feats)
        self.time2v = Time2Vec('sin',out_feats)

    # def forward(self, graph, memory, ts, base_embedding):
    #     graph = graph.local_var()
    #     graph.ndata['timestamp'] = ts
    #     efeat = self.preprocessor(graph).float()
    #     # rst = memory
    #     base_hid = F.relu(self.base_fc(base_embedding))
    #     memory_hid = F.relu(self.memory_fc(memory))
    #     rst = F.relu(self.emb_fc(base_hid+memory_hid))
    #     for i in range(self.layers-1):
    #         rst = self.layer_list[i](graph, rst, efeat).flatten(1)
    #     # print(temp)
    #     # rst = self.layer_list[-1](graph, rst, efeat).mean(1)
    #     rst = self.layer_list[-1](graph, rst, efeat)

    #     # memory_decode = F.relu(self.memory_decode_fc(rst))
    #     # print(rst)
    #     # exit()
    #     return rst, rst

    def forward(self, graph, memory, ts, base_embedding, cat_relation_graph, cat_embedding):
        graph = graph.local_var()
        graph.ndata['timestamp'] = ts
        # efeat = self.preprocessor(graph).float()
        efeat = graph.edata['feats_type']
        # rst = memory
        # time_encode = self.time2v(ts.unsqueeze(dim=1).float())
        time_encode = self.time2v(graph.edata['timestamp'].unsqueeze(dim=1).float())
        # print(time_encode)
        # print(time_encode.shape)
        # exit()
        # time_encode = self.temporal_encoder(
        #     ts.unsqueeze(dim=1)).view(ts.shape[0], -1)
        # # memory_feat = torch.cat([memory, time_encode], dim=1)
        graph.edata['time_emb'] = time_encode
        rst = base_embedding
        cat_relation_graph = cat_relation_graph.local_var()
        # cat_relation_graph.ndata['emb'] = cat_embedding
        base_hid,cat_hid = self.layer_list[-1](graph, rst, efeat, cat_relation_graph, cat_embedding)
        # memory_hid = F.relu(self.memory_fc(memory.float()))
        # rst = F.relu(self.emb_fc(base_hid+memory_hid))
        # memory_decode = F.relu(self.memory_decode_fc(rst))
        # print(rst)
        # exit()
        return base_hid,base_hid,cat_hid


def t2v(tau, f, out_features, w, b, w0, b0, arg = None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1)) #rand是生成数符合均匀分布[0,1) randn符合标准正态分布
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin
    
    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        else:
            self.l1 = CosineActivation(1, out_dim)
        
    def forward(self, x):
        x = self.l1(x)
        return x