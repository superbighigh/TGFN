import torch
from torch.utils.data import DataLoader
import numpy as np
import time, os
import pickle
from setting import Setting
from trainer import FlashbackTrainer
from dataloader import PoiDataloader
from dataset import Split
from utils import *
from network import create_h0_strategy
from evaluation import Evaluation
from tqdm import tqdm
from scipy.sparse import coo_matrix
import dgl

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# parse settings
setting = Setting()
setting.parse()
dir_name = os.path.dirname(setting.log_file)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
setting.log_file = setting.log_file + '_' + timestring
log = open(setting.log_file, 'w')


message = ''.join([f'{k}: {v}\n' for k, v in vars(setting).items()])
log_string(log, message)

# load dataset
poi_loader = PoiDataloader(setting,
    setting.max_users, setting.min_checkins)  # 0， 5*20+1
poi_loader.read(setting.dataset_file)
# print('Active POI number: ', poi_loader.locations())  # 18737 106994
# print('Active User number: ', poi_loader.user_count())  # 32510 7768
# print('Total Checkins number: ', poi_loader.checkins_count())  # 1278274

log_string(log, 'Active POI number:{}'.format(poi_loader.locations()))
log_string(log, 'Active User number:{}'.format(poi_loader.user_count()))
log_string(log, 'Total Checkins number:{}'.format(poi_loader.checkins_count()))

dataset = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TRAIN, setting.sampler_days, setting.fri_sampler_count, setting.loc_sampler_count)  # 20, 200 or 1024, 0
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
dataset_test = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TEST, setting.sampler_days, setting.fri_sampler_count,setting.loc_sampler_count)
dataset_test.total_g = dataset.total_g
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
# print(dataset_test)
# print(dataloader_test.dataset)
# exit()
assert setting.batch_size < poi_loader.user_count(
), 'batch size must be lower than the amount of available users'

# create flashback trainer

transition_graph = None
spatial_graph = None
friend_graph = None
interact_graph = None
# with open(setting.trans_loc_file, 'rb') as f:  # transition POI graph
#     transition_graph = pickle.load(f)  # 在cpu上
# transition_graph = coo_matrix(transition_graph)
# with open(setting.trans_interact_file, 'rb') as f:  # User-POI interaction graph
#     interact_graph = pickle.load(f)  # 在cpu上
# interact_graph = csr_matrix(interact_graph)

log_string(log, 'Successfully load graph')

trainer = FlashbackTrainer(setting.lambda_t, setting.lambda_s, setting.lambda_loc, setting.lambda_user,
                           setting.use_weight, transition_graph, spatial_graph, friend_graph, setting.use_graph_user,
                           setting.use_spatial_graph, interact_graph, setting.sq_count, setting.sp_count, setting, poi_loader.cat_g,poi_loader.cat_relation_graph)  # 0.01, 100 or 1000
h0_strategy = create_h0_strategy(
    setting.hidden_dim, setting.is_lstm)  # 10 True or False
trainer.prepare(poi_loader.locations(), poi_loader.user_count(), poi_loader.cat_count, setting.hidden_dim, setting.rnn_factory,
                setting.device)
evaluation_test = Evaluation(dataset_test, dataloader_test,
                             poi_loader.user_count(), h0_strategy, trainer, setting, log)
print('{} {}'.format(trainer, setting.rnn_factory))

#  training loop
optimizer = torch.optim.Adam(trainer.parameters(
), lr=setting.learning_rate, weight_decay=setting.weight_decay)
if setting.scheduler_type == "normal":
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 50], gamma=0.2)
elif setting.scheduler_type == "15":
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45, 50], gamma=0.2)
else:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 50], gamma=0.2)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(
#     optimizer, milestones=[10, 20, 30, 40, 50], gamma=0.2)

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.2)
param_count = trainer.count_parameters()
log_string(log, f'In total: {param_count} trainable parameters')

bar = tqdm(total=setting.epochs)
bar.set_description('Training')

for e in range(setting.epochs):  # 100
    h = h0_strategy.on_init(setting.batch_size, setting.device)
    dataset.shuffle_users()  # shuffle users before each epoch!

    losses = []
    epoch_start = time.time()
    for i, (x, t, t_slot, s, y, y_t, y_t_slot, y_s, reset_h, active_users, t_hour, cat) in enumerate(dataloader):
        # reset hidden states for newly added users
        # if i == 10:
        #     exit()
        # print("batch sta", i, time.time())
        for j, reset in enumerate(reset_h):
            if reset:
                if setting.is_lstm:
                    hc = h0_strategy.on_reset(active_users[0][j])
                    h[0][0, j] = hc[0]
                    h[1][0, j] = hc[1]
                else:
                    h[0, j] = h0_strategy.on_reset(active_users[0][j])

        x = x.squeeze().to(setting.device)
        t = t.squeeze().to(setting.device)
        t_slot = t_slot.squeeze().to(setting.device)
        t_hour = t_hour.squeeze().to(setting.device)
        s = s.squeeze().to(setting.device)
        cat = cat.squeeze().to(setting.device)

        y = y.squeeze().to(setting.device)
        y_t = y_t.squeeze().to(setting.device)
        y_t_slot = y_t_slot.squeeze().to(setting.device)
        y_s = y_s.squeeze().to(setting.device)
        active_users = active_users.to(setting.device)
        # positive_pair_g = dataloader.dataset.positive_pair_g
        # g = dataloader.dataset.block
        positive_pair_g = dataloader.dataset.positive_pair_g.to(setting.device)
        # 正负样本
        negative_pair_g = dataloader.dataset.negative_pair_g.to(setting.device)
        # 正负样本
        g = dataloader.dataset.block.to(setting.device)
        # cat_g = dataloader.dataset.cat_g.to(setting.device)
        node_index_key = dataloader.dataset.node_index_key
        # print(x)
        
        optimizer.zero_grad()
        loss = trainer.loss(x, t, t_slot, s, y, y_t,
                            y_t_slot, y_s, h, active_users, g, positive_pair_g, negative_pair_g, t_hour,node_index_key, cat)
        # print("get loss", time.time())
        loss.backward(retain_graph=True)
        # for name, param in trainer.model.named_parameters():
        #     print("Parameter name:", name,param.grad is None)
        #     print("Gradient:", )
        # torch.nn.utils.clip_grad_norm_(trainer.parameters(), 5)
        losses.append(loss.item())
        optimizer.step()
        # p_u = trainer.model.encoder(active_users)  # (1, user_len, hidden_size)
        # print(p_u)
        # emb_memory = trainer.model.memory.memory[g.ndata[dgl.NID].to('cpu'), :]
        # emb_t = g.ndata['timestamp']
        # embedding = trainer.model.embedding_attn(g, emb_memory, emb_t)
        # emb2pred = dict(
        #     zip(g.ndata[dgl.NID].tolist(), g.nodes().tolist()))
        # print(embedding[0])
        # print("loss backward", time.time())
        # trainer.detach_memory()
        # trainer.update_memory(g)
        # emb_memory = trainer.model.memory.memory[g.ndata[dgl.NID].to('cpu'), :]
        # emb_t = g.ndata['timestamp']
        # embedding = trainer.model.embedding_attn(g, emb_memory, emb_t)
        # emb2pred = dict(
        #     zip(g.ndata[dgl.NID].tolist(), g.nodes().tolist()))
        # print(embedding[0])
        # memory_checkpoint = trainer.store_memory()
        # trainer.restore_memory(memory_checkpoint)
        # print("batch end", i, time.time())
        # exit()
        # if i>=30:
        #     exit()
    # schedule learning rate:
    scheduler.step()
    bar.update(1)
    epoch_end = time.time()
    log_string(log, 'One training need {:.2f}s'.format(
        epoch_end - epoch_start))
    # statistics:
    if (e + 1) % 1 == 0:
        epoch_loss = np.mean(losses)
        log_string(log, f'Epoch: {e + 1}/{setting.epochs}')
        log_string(log, f'Used learning rate: {scheduler.get_last_lr()[0]}')
        log_string(log, f'Avg Loss: {epoch_loss}')

    if (e + 1) % setting.validate_epoch == 0:
        log_string(log, f'~~~ Test Set Evaluation (Epoch: {e + 1}) ~~~')
        evl_start = time.time()
        evaluation_test.dataloader.dataset.reset_graph = 0
        evaluation_test.dataloader.dataset.g = dataloader.dataset.g
        evaluation_test.evaluate(e+1)
        evl_end = time.time()
        log_string(log, 'One evaluate need {:.2f}s'.format(
            evl_end - evl_start))

        if (e+1) == 30 and setting.model_name!='no':
            trainer.model.cpu()
            h.cpu()
            torch.save({'model':trainer.model.state_dict(),'h':h},'results/pre_model_{}.ckpt'.format(setting.model_name))
            trainer.model.cuda()
            h.cuda()

    trainer.reset_memory()
bar.close()
