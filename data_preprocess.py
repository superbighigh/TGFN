import os
import ssl
from six.moves import urllib

import pandas as pd
import numpy as np

import torch
import dgl
from setting import Setting
from dataloader import PoiDataloader

# === Below data preprocessing code are based on
# https://github.com/twitter-research/tgn

# Preprocess the raw data split each features

def preprocess(data_name,friend_path,loc_path):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []
    user2id = {}
    poi2id = {}
    pre_user = -1
    pre_i = -1
    e_feat = torch.randn(4, 10).numpy()
    index = 0
    with open(data_name) as f:
        # s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split('\t')
            u = int(e[0])
            i = int(e[4])
            
            user2id[prev_user] = len(self.user2id)
            
            ts = (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ") - datetime(1970, 1, 1)).total_seconds()

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            idx_list.append(index)
            index+=1
            feat_l.append(e_feat[0])


            if pre_i != -1 and u == pre_user:
                u_list.append(pre_i)
                i_list.append(i)
                ts_list.append(ts)
                idx_list.append(index)
                index+=1
                feat_l.append(e_feat[1])

            if pre_i == -1:
                pre_i = i

            if pre_user != u:
                pre_user = u
    min_ts = data.ts.min()
    data.ts -= min_ts
    data = pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'idx': idx_list}) 
    data = data.sort_values(by='ts')
    data = data.reset_index(drop=True)
    u_list, i_list, ts_list, feat_loc, idx_list = [],[],[],[],[]
    with open(loc_path) as f:
        for idx, line in enumerate(f):
            e = line.strip().split('\t')
            u = int(e[0])
            i = int(e[1])
            u_list.append(u)
            i_list.append(i)
            ts_list.append(0)
            idx_list.append(idx)
            feat_loc.append(e_feat[2])
    data_loc = pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'idx': idx_list})  
    u_list, i_list, ts_list, feat_fri, idx_list = [],[],[],[],[]
    with open(friend_path) as f:
        for idx, line in enumerate(f):
            e = line.strip().split('\t')
            u = int(e[0])
            i = int(e[1])
            u_list.append(u)
            i_list.append(i)
            ts_list.append(0)
            idx_list.append(idx)
            feat_fri.append(e_feat[3])
    data_fri = pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'idx': idx_list})    
    
    return data,feat_l,data_loc,feat_loc,data_fri,feat_fri

# Re index nodes for DGL convience
def reindex(df, bipartite=True):
    new_df = df.copy()
    if bipartite:
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    else:
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

    return new_df

def loadData(loc_path,friend_path):
    e_feat = torch.randn(4, 10).numpy()
    poi_loader = PoiDataloader(setting.max_users, setting.min_checkins)  # 0， 5*20+1
    poi_loader.read(setting.dataset_file)
    print('Active POI number: ', poi_loader.locations())  # 18737
    print('Active User number: ', poi_loader.user_count())  # 32510
    print('Total Checkins number: ', poi_loader.checkins_count())  # 1278274
    users_count = len(users)
    users = poi_loader.users
    pois = poi_loader.locs
    times = poi_loader.times
    poi2id = poi_loader.poi2id
    user2id = poi_loader.user2id
    u_list, i_list, ts_list, feat_l, idx_list, tag_list = [], [], [], [], [], []
    index = 0
    for i, (user, loc, time) in enumerate(zip(users, pois, tiems)):
        train_thr = int(len(loc) * 0.8)
        train_locs = loc[: train_thr]
        train_time = time[:train_thr]
        test_locs = loc[train_thr:]
        test_time = time[train_thr:]
        pre_poi = -1
        for j in range(len(loc)):
            poi = int(loc[j]) + users_count
            t = time[j]
            u_list.append(int(user))
            i_list.append(int(poi))
            ts_list.append(t)
            feat_l.append(e_feat[0])
            idx_list.append(index)
            index+=1
            if j < train_thr:
                tag_list.append(0)
            else:
                tag_list.append(1)
            if pre_poi != -1 and j != train_thr:
                u_list.append(int(pre_loc))
                i_list.append(int(poi))
                ts_list.append(t)
                feat_l.append(e_feat[1])
                idx_list.append(index)
                index+=1
                if j < train_thr:
                    tag_list.append(0)
                else:
                    tag_list.append(1)
            pre_poi = poi


    data = pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'tag': tag_list
                         'idx': idx_list}) 
    min_ts = data.ts.min()
    data.ts -= min_ts
    data = data.sort_values(by='ts')
    data = data.reset_index(drop=True)

    u_list, i_list, ts_list, feat_loc, idx_list = [],[],[],[],[]
    with open(loc_path) as f:
        for idx, line in enumerate(f):
            e = line.strip().split('\t')
            u = int(e[0])
            i = int(e[1])
            u_list.append(u)
            i_list.append(i)
            ts_list.append(0)
            idx_list.append(idx)
            feat_loc.append(e_feat[2])
    data_loc = pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'idx': idx_list}) 
    u_list, i_list, ts_list, feat_fri, idx_list = [],[],[],[],[]
    with open(friend_path) as f:
        for idx, line in enumerate(f):
            e = line.strip().split('\t')
            u = int(e[0])
            i = int(e[1])
            if user2id.get(u) and user2id.get(i):  # only focus on active users
                user_id1 = int(user2id.get(u))
                user_id2 = int(user2id.get(i))
                u_list.append(user_id1)
                u_list.append(user_id2)
                i_list.append(user_id2)
                i_list.append(user_id1)
                ts_list.append(0)
                ts_list.append(0)
                idx_list.append(idx*2)
                idx_list.append(idx*2+1)
                feat_fri.append(e_feat[3])
                feat_fri.append(e_feat[3])
    data_fri = pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'idx': idx_list})
    return data,feat_l,data_loc,feat_loc,data_fri,feat_fri


# Save edge list, features in different file for data easy process data
def run(data_name, bipartite=True):
    PATH = './data/checkins-{}.txt'.format(data_name)
    friend_path = './data/{}_friend.txt'.format(data_name)
    loc_path = './data/{}_poi_spatial_triplets.txt'.format(data_name)
    OUT_DF = './data/ml_{}.csv'.format(data_name)
    OUT_DF_LOC = './data/ml_loc_{}.csv'.format(data_name)
    OUT_DF_FRI = './data/ml_fri_{}.csv'.format(data_name)
    OUT_FEAT = './data/ml_{}.npy'.format(data_name)
    OUT_FEAT_LOC = './data/ml_loc_{}.npy'.format(data_name)
    OUT_FEAT_FRI = './data/ml_fri_{}.npy'.format(data_name)
    # OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

    # df, feat, df_loc, feat_loc, df_fri, feat_fri = preprocess(PATH,friend_path,loc_path)
    df, feat, df_loc, feat_loc, df_fri, feat_fri = loadData(friend_path,loc_path)
    new_df = reindex(df, False)
    df_loc = reindex(df_loc, False)
    df_fri = reindex(df_fri, False)

    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat])
    feat_loc = np.vstack([empty, feat_loc])
    feat_fri = np.vstack([empty, feat_fri])

    # max_idx = max(new_df.u.max(), new_df.i.max())
    # rand_feat = np.zeros((max_idx + 1, 172))

    new_df.to_csv(OUT_DF)
    df_loc.to_csv(OUT_DF_LOC)
    df_fri.to_csv(OUT_DF_FRI)
    np.save(OUT_FEAT, feat)
    np.save(OUT_FEAT_LOC, feat_loc)
    np.save(OUT_FEAT_FRI, feat_fri)
    # np.save(OUT_NODE_FEAT, rand_feat)

# === code from twitter-research-tgn end ===

# If you have new dataset follow by same format in Jodie,
# you can directly use name to retrieve dataset

def loadGraph(link_path,feat_path):
    raw_connection = pd.read_csv(link_path)
    raw_feature = np.load(feat_path)
    # -1 for re-index the node
    src = raw_connection['u'].to_numpy()-1
    dst = raw_connection['i'].to_numpy()-1
    # Create directed graph
    g = dgl.graph((src, dst))
    g.edata['timestamp'] = torch.from_numpy(
        raw_connection['ts'].to_numpy())
    g.edata['tag'] = torch.from_numpy(raw_connection['tag'].to_numpy())
    # g.edata['label'] = torch.from_numpy(raw_connection['label'].to_numpy())
    g.edata['feats'] = torch.from_numpy(raw_feature[1:, :]).float()
    return g

def TemporalDataset(dataset):
    if not os.path.exists('./data/{}.bin'.format(dataset)):
        # if not os.path.exists('./data/{}.csv'.format(dataset)):
        #     if not os.path.exists('./data'):
        #         os.mkdir('./data')

        #     url = 'https://snap.stanford.edu/jodie/{}.csv'.format(dataset)
        #     print("Start Downloading File....")
        #     context = ssl._create_unverified_context()
        #     data = urllib.request.urlopen(url, context=context)
        #     with open("./data/{}.csv".format(dataset), "wb") as handle:
        #         handle.write(data.read())

        print("Start Process Data ...")
        run(dataset)
        
        g = loadGraph('./data/ml_{}.csv'.format(dataset), './data/ml_{}.npy'.format(dataset))
        g_loc = loadGraph('./data/ml_loc_{}.csv'.format(dataset), './data/ml_loc_{}.npy'.format(dataset))
        g_fri = loadGraph('./data/ml_fri_{}.csv'.format(dataset), './data/ml_fri_{}.npy'.format(dataset))
        
        dgl.save_graphs('./data/{}.bin'.format(dataset), [g,g_loc,g_fri])
    else:
        print("Data is exist directly loaded.")
        gs, _ = dgl.load_graphs('./data/{}.bin'.format(dataset))
        g = gs[0]
        g_loc = gs[1]
        g_fri = gs[2]
    return g, g_loc, g_fri

def TemporalWikipediaDataset():
    # Download the dataset
    return TemporalDataset('wikipedia')

def TemporalRedditDataset():
    return TemporalDataset('reddit')

