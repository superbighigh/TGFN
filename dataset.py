import random
from enum import Enum
import torch
from torch.utils.data import Dataset
import dgl
import numpy as np
from functools import partial
import time, os

class Split(Enum):
    """ Defines whether to split for train or test.
    """
    TRAIN = 0
    TEST = 1


class Usage(Enum):
    """
    Each user has a different amount of sequences. The usage defines
    how many sequences are used:

    MAX: each sequence of any user is used (default)
    MIN: only as many as the minimal user has
    CUSTOM: up to a fixed amount if available.

    The unused sequences are discarded. This setting applies after the train/test split.
    """

    MIN_SEQ_LENGTH = 0
    MAX_SEQ_LENGTH = 1
    CUSTOM = 2


class PoiDataset(Dataset):
    """
    Our Point-of-interest pytorch dataset: To maximize GPU workload we organize the data in batches of
    "user" x "a fixed length sequence of locations". The active users have at least one sequence in the batch.
    In order to fill the batch all the time we wrap around the available users: if an active user
    runs out of locations we replace him with a new one. When there are no unused users available
    we reuse already processed ones. This happens if a single user was way more active than the average user.
    The batch guarantees that each sequence of each user was processed at least once.

    This data management has the implication that some sequences might be processed twice (or more) per epoch.
    During training you should call PoiDataset::shuffle_users before the start of a new epoch. This
    leads to more stochastic as different sequences will be processed twice.
    During testing you *have to* keep track of the already processed users.

    Working with a fixed sequence length omits awkward code by removing only few of the latest checkins per user. We
    work with a 80/20 train/test spilt, where test check-ins are strictly after training checkins. To obtain at least
    one test sequence with label we require any user to have at least (5*<sequence-length>+1) checkins in total.
    """

    def reset(self):
        # reset training state:
        self.next_user_idx = 0  # current user index to add
        self.active_users = []  # current active users
        self.active_user_seq = []  # current active users sequences
        # self.user_permutation = []  # shuffle users during training
        self.all_user_seq = []
        # set active users:
        for i in range(self.batch_size):
            self.active_users.append(i)

        for i in range(len(self.users)):  # 200 or 1024
            # self.next_user_idx = (self.next_user_idx + 1) % len(self.users)  # 200 or 1024
            # self.active_users.append(i)  # [0, 1, ..., 199] or [0, 1, ..., 1023]
            self.all_user_seq.append(0)  # [0, 0, ..., 0] start from user's first subsequence

        # # use 1:1 permutation:
        # for i in range(len(self.users)):
        #     self.user_permutation.append(i)  # [0, 1, ..., User_NUM-1]

    def shuffle_users(self):
        # random.shuffle(self.user_permutation)  # 原地随机打乱

        # reset active users:
        self.next_user_idx = 0
        self.active_users = []
        # self.active_user_seq = []
        self.all_user_seq = []
        self.reset_graph = 1
        total_nodes = len(self.users) + self.loc_count
        # self.g = dgl.graph(([], []), num_nodes=total_nodes)
        # self.g.ndata[dgl.NID] = torch.arange(self.g.num_nodes())
        # print(self.g.nodes())
        # print("graph reset", self.split)

        for i in range(self.batch_size):
            self.active_users.append(i)

        for i in range(len(self.users)):
            # self.next_user_idx = (self.next_user_idx + 1) % len(self.users)  # 200(1024),下一活跃用户也即打乱后的第201个用户
            # self.active_users.append(self.user_permutation[i])  # 活跃用户为打乱后的前200个(1024)用户
            self.all_user_seq.append(0)

    def __init__(self, users, times, time_slots, coords, locs, sequence_length, batch_size, split, usage, loc_count, custom_seq_count, loc_g, fri_g, cat_g, days, fri_sampler_count,loc_sampler_count, weekdays, months, hours, hour_g, categorys,setting):
        self.users = users
        self.locs = locs
        self.times = times
        self.time_slots = time_slots
        self.coords = coords
        self.loc_g = loc_g
        self.fri_g = fri_g
        self.cat_g = cat_g
        self.hour_g = hour_g
        self.weekdays = weekdays
        self.months = months
        self.hours = hours
        self.categorys = categorys
        self.fri_sampler = partial(dgl.sampling.sample_neighbors, fanout=fri_sampler_count)
        self.loc_sampler = partial(dgl.sampling.sample_neighbors, fanout=loc_sampler_count)
        self.sampler = partial(dgl.sampling.select_topk, k=-1, weight='timestamp')
        self.neg_sampler = dgl.dataloading.negative_sampler.Uniform(k=1)
        self.setting = setting


        self.labels = []  # 二重列表
        self.lbl_times = []  # labels所对应的GPS坐标以及timestamp
        self.lbl_time_slots = []
        self.lbl_coords = []

        self.sequences = []  # 三重列表,每个元素表示以固定长度(20)子序列的形式来存放每个用户的check-ins
        self.sequences_times = []
        self.sequences_time_slots = []
        self.sequences_coords = []
        self.sequences_categorys = []

        self.sequences_labels = []
        self.sequences_lbl_times = []
        self.sequences_lbl_time_slots = []
        self.sequences_lbl_coords = []
        self.sequences_weekdays = []
        self.sequences_months = []
        self.sequences_hours = []

        self.sequences_count = []  # 存放每个用户的子序列的数目
        self.Ps = []  # 没用?
        self.Qs = torch.zeros(loc_count, 1)  # 有什么用?
        self.usage = usage  # MAX_SEQ_LENGTH
        self.batch_size = batch_size  # 200 or 1024
        self.loc_count = loc_count
        self.custom_seq_count = custom_seq_count  # 1
        self.user_seq_times = []
        self.user_permutation = []
        self.edge_feats = torch.randn(4, 10).numpy()
        self.negative_pair_g = dgl.graph(([], []))
        self.days = days

        self.reset()

        src_list = []
        dst_list = []
        ts_list = []
        feat_list = []
        feat_type = []
        weekday_list = []
        month_list = []
        hour_list = []

        # collect locations:
        for i in range(loc_count):
            self.Qs[i, 0] = i

        # align labels to locations (shift by one)
        for i, loc in enumerate(locs):
            self.locs[i] = loc[:-1]  # data,列表
            self.labels.append(loc[1:])  # labels,列表
            self.lbl_times.append(self.times[i][1:])
            self.lbl_time_slots.append(self.time_slots[i][1:])
            self.lbl_coords.append(self.coords[i][1:])

            self.times[i] = self.times[i][:-1]
            self.time_slots[i] = self.time_slots[i][:-1]
            self.coords[i] = self.coords[i][:-1]
            self.weekdays[i] = self.weekdays[i][:-1]
            self.months[i] = self.months[i][:-1]
            self.hours[i] = self.hours[i][:-1]


        # split to training / test phase:
        for i, (time, time_slot, coord, loc, label, lbl_time, lbl_time_slot, lbl_coord, weekdays, months, hours, categorys) in enumerate(
                zip(self.times, self.time_slots, self.coords, self.locs, self.labels, self.lbl_times,
                    self.lbl_time_slots, self.lbl_coords, self.weekdays, self.months, self.hours, self.categorys)):
            train_thr = int(len(loc) * 0.8)
            if split == Split.TRAIN:
                self.locs[i] = loc[:train_thr]
                self.times[i] = time[:train_thr]
                self.time_slots[i] = time_slot[:train_thr]
                self.coords[i] = coord[:train_thr]

                self.labels[i] = label[:train_thr]
                self.lbl_times[i] = lbl_time[:train_thr]
                self.lbl_time_slots[i] = lbl_time_slot[:train_thr]
                self.lbl_coords[i] = lbl_coord[:train_thr]

                self.weekdays[i] = weekdays[:train_thr]
                self.months[i] = months[:train_thr]
                self.hours[i] = hours[:train_thr]
                self.categorys[i] = categorys[:train_thr]

                for k in range(len(self.locs[i])):
                    # src_list.append(int(i))
                    # dst_list.append(int(self.locs[i][k])+len(self.users))
                    # ts_list.append(self.times[i][k])
                    # feat_list.append(self.edge_feats[0])
                    # weekday_list.append(self.weekdays[i][k])
                    # month_list.append(self.months[i][k])
                    # hour_list.append(self.hours[i][k])
                    # feat_type.append(0)
                    dst_list.append(int(i))
                    src_list.append(int(self.locs[i][k])+len(self.users))
                    ts_list.append(self.times[i][k])
                    feat_list.append(self.edge_feats[0])
                    feat_type.append(0)
                    # weekday_list.append(self.weekdays[i][k])
                    # month_list.append(self.months[i][k])
                    # hour_list.append(self.hours[i][k])
                    if k+1 < len(self.locs[i]):
                        src_list.append(int(self.locs[i][k])+len(self.users))
                        dst_list.append(int(self.locs[i][k+1])+len(self.users))
                        ts_list.append(self.times[i][k])
                        feat_list.append(self.edge_feats[1])
                        feat_type.append(1)
                        weekday_list.append(self.weekdays[i][k])
                        month_list.append(self.months[i][k])
                        hour_list.append(self.hours[i][k])
                        # dst_list.append(int(self.locs[i][k])+len(self.users))
                        # src_list.append(int(self.locs[i][k+1])+len(self.users))
                        # ts_list.append(self.times[i][k])
                        # feat_list.append(self.edge_feats[1])
                        # feat_type.append(1)
                        # weekday_list.append(self.weekdays[i][k+1])
                        # month_list.append(self.months[i][k+1])
                        # hour_list.append(self.hours[i][k+1])

            if split == Split.TEST:
                if self.setting.test_model == 'no':
                    self.locs[i] = loc[train_thr:]
                    self.times[i] = time[train_thr:]
                    self.time_slots[i] = time_slot[train_thr:]
                    self.coords[i] = coord[train_thr:]

                    self.labels[i] = label[train_thr:]
                    self.lbl_times[i] = lbl_time[train_thr:]
                    self.lbl_time_slots[i] = lbl_time_slot[train_thr:]
                    self.lbl_coords[i] = lbl_coord[train_thr:]
                    self.weekdays[i] = weekdays[train_thr:]
                    self.months[i] = months[train_thr:]
                    self.hours[i] = hours[train_thr:]
                    self.categorys[i] = categorys[train_thr:]
                else:
                    temp_loc,temp_time,temp_time_slot,temp_coord,temp_weekdays,temp_months,temp_hours,temp_categorys = [],[],[],[],[],[],[],[]
                    for test_i in range(20):
                        temp_loc.append(63)
                        temp_time.append(1800*i)
                        temp_time_slot.append(0)
                        temp_coord.append((37.61635606, -122.38615036))
                        temp_weekdays.append(0)
                        temp_months.append(0)
                        temp_hours.append(0)
                        temp_categorys.append(0)
                    self.locs[i] = temp_loc
                    self.times[i] = temp_time
                    self.time_slots[i] = temp_time_slot
                    self.coords[i] = temp_coord

                    self.labels[i] = temp_loc
                    self.lbl_times[i] = temp_time
                    self.lbl_time_slots[i] = temp_time_slot
                    self.lbl_coords[i] = temp_coord
                    self.weekdays[i] = temp_weekdays
                    self.months[i] = temp_months
                    self.hours[i] = temp_hours
                    self.categorys[i] = temp_categorys



        ts_list = torch.tensor(ts_list,dtype=torch.double,requires_grad=False)
        feat_list = torch.from_numpy(np.array(feat_list))
        feat_type_list = torch.from_numpy(np.array(feat_type))
        weekday_list_1 = torch.from_numpy(np.array(weekday_list))
        month_list_1 = torch.from_numpy(np.array(month_list))
        hour_list_1 = torch.from_numpy(np.array(hour_list))
        total_nodes = len(self.users) + self.loc_count

        if split == Split.TRAIN:
            unique_edges = set(zip(src_list, dst_list))
            unique_src, unique_dst = zip(*unique_edges)
            self.g = dgl.graph((torch.tensor(unique_src), torch.tensor(unique_dst)),num_nodes=total_nodes)
            self.g.ndata[dgl.NID] = torch.arange(self.g.num_nodes())
            self.g.edata['feats_type'] = torch.from_numpy(np.ones(self.g.num_edges(),dtype=np.int64 )*1)
            self.g.edata['timestamp'] = torch.from_numpy(np.zeros(self.g.num_edges(),dtype=np.double ))
            # self.total_g = dgl.merge([self.g,self.loc_g,self.cat_g])
            if self.setting.relation_type_select == "all":
                self.total_g = dgl.merge([self.g,self.loc_g])
            elif self.setting.relation_type_select == "sp":
                self.total_g = self.loc_g 
            elif self.setting.relation_type_select == "sq":
                self.total_g = self.g

        # split location and labels to sequences:
        self.max_seq_count = 0
        self.min_seq_count = 10000000
        self.capacity = 0
        for i, (time, time_slot, coord, loc, label, lbl_time, lbl_time_slot, lbl_coord, weekdays, months, hours, categorys) in enumerate(
                zip(self.times, self.time_slots, self.coords, self.locs, self.labels, self.lbl_times,
                    self.lbl_time_slots, self.lbl_coords, self.weekdays, self.months, self.hours, self.categorys)):
            seq_count = len(loc) // sequence_length  # 统计每个用户的sequence数目, 至少5个
            assert seq_count > 0, 'fix seq-length and min-checkins in order to have at least one test sequence in a 80/20 split!'
            seqs = []  # 二重列表,以固定长度子序列的形式来存放每个用户的check-ins
            seq_times = []
            seq_time_slots = []
            seq_coords = []

            seq_lbls = []
            seq_lbl_times = []
            seq_lbl_time_slots = []
            seq_lbl_coords = []
            seq_weekdays = []
            seq_months = []
            seq_hours = []
            seq_categorys = []
            # user_seq_times = []

            for j in range(seq_count):
                start = j * sequence_length
                end = (j + 1) * sequence_length
                seqs.append(loc[start:end])
                seq_times.append(time[start:end])
                self.user_seq_times.append([i, max(time[start:end])])
                seq_time_slots.append(time_slot[start:end])
                seq_coords.append(coord[start:end])
                seq_categorys.append(categorys[start:end])

                seq_lbls.append(label[start:end])
                seq_lbl_times.append(lbl_time[start:end])
                seq_lbl_time_slots.append((lbl_time_slot[start:end]))
                seq_lbl_coords.append(lbl_coord[start:end])
                seq_weekdays.append(weekdays[start:end])
                seq_months.append(months[start:end])
                seq_hours.append(hours[start:end])

            self.sequences.append(seqs)
            self.sequences_times.append(seq_times)
            self.sequences_time_slots.append(seq_time_slots)
            self.sequences_coords.append(seq_coords)
            self.sequences_categorys.append(seq_categorys)

            self.sequences_labels.append(seq_lbls)
            self.sequences_lbl_times.append(seq_lbl_times)
            self.sequences_lbl_time_slots.append(seq_lbl_time_slots)
            self.sequences_lbl_coords.append(seq_lbl_coords)
            self.sequences_weekdays.append(seq_weekdays)
            self.sequences_months.append(seq_months)
            self.sequences_hours.append(seq_hours)

            self.sequences_count.append(seq_count)
            self.capacity += seq_count
            self.max_seq_count = max(self.max_seq_count, seq_count)
            self.min_seq_count = min(self.min_seq_count, seq_count)

        self.user_seq_times.sort(key=lambda x: x[1])
        self.user_permutation = [row[0] for row in self.user_seq_times]
        self.split = split

        # statistics
        if self.usage == Usage.MIN_SEQ_LENGTH:
            print(split, 'load', len(users), 'users with min_seq_count', self.min_seq_count, 'batches:', self.__len__())
        if self.usage == Usage.MAX_SEQ_LENGTH:
            print(split, 'load', len(users), 'users with max_seq_count', self.max_seq_count, 'batches:', self.__len__())
        if self.usage == Usage.CUSTOM:
            print(split, 'load', len(users), 'users with custom_seq_count', self.custom_seq_count, 'Batches:',
                  self.__len__())

    # def construct_graph():
    #     src_list, dst_list, ts_list, feat_type = [], [], [], []
    #     for i_user in self.user_permutation:
    #         j = self.all_user_seq[i_user]  # 0
    #         max_j = self.sequences_count[i_user]  # 用户i所拥有的序列数目
    #         for k in range(20):
    #             src_list.append(int(self.sequences[i_user][j][k])+len(self.users))
    #             dst_list.append(int(self.sequences_labels[i_user][j][k])+len(self.users))
    #             ts_list.append(self.sequences_times[i_user][j][k])
    #             # feat_list.append(self.edge_feats[1])
    #             feat_type.append(1)
    #         self.all_user_seq[i_user] += 1
    #     total_nodes = len(self.users) + self.loc_count
    #     ts_list = torch.tensor(ts_list,dtype=torch.double,requires_grad=False)
    #     self.g = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)),num_nodes=total_nodes)
    #     self.g.ndata[dgl.NID] = torch.arange(self.g.num_nodes())
    #     self.g.edata['feats_type'] = torch.from_numpy(np.ones(self.g.num_edges(),dtype=np.int64 )*1)
    #     self.g.edata['timestamp'] = ts_list
    #     self.all_user_seq = []
    #     for i in range(len(self.users)):  # 200 or 1024
    #         self.all_user_seq.append(0)

    def sequences_by_user(self, idx):
        return self.sequences[idx]

    def __len__(self):
        """ Amount of available batches to process each sequence at least once.
        """

        if self.usage == Usage.MIN_SEQ_LENGTH:
            # min times amount_of_user_batches:
            return self.min_seq_count * (len(self.users) // self.batch_size)
        if self.usage == Usage.MAX_SEQ_LENGTH:
            # estimated capacity:
            estimated = self.capacity // self.batch_size
            return max(self.max_seq_count, estimated)
        if self.usage == Usage.CUSTOM:
            return self.custom_seq_count * (len(self.users) // self.batch_size)
        raise ValueError()

    def __getitem__(self, idx):
        """ Against pytorch convention, we directly build a full batch inside __getitem__.
        Use a batch_size of 1 in your pytorch data loader.

        A batch consists of a list of active users,
        their next location sequence with timestamps and coordinates.

        y is the target location and y_t, y_s the targets timestamp and coordinates. Provided for
        possible use.

        reset_h is a flag which indicates when a new user has been replacing a previous user in the
        batch. You should reset this users hidden state to initial value h_0.
        """
        # print("get data start", time.time())

        seqs = []
        times = []
        time_slots = []
        time_hours = []
        coords = []
        categorys = []

        lbls = []
        lbl_times = []
        lbl_time_slots = []
        lbl_coords = []

        reset_h = []
        loc_trans_count = {}
        user_loc_count = {}
        # print("idx:{}, len:{}".format(idx, len(self.user_permutation)))
        src_list = []
        dst_list = []
        ts_list = []
        feat_list = []
        feat_type = []
        weekday_list = []
        month_list = []
        hour_list = []
        node_index_key = {}
        node_repeat_array = {}
        node_repeat_time = {}
        node_repeat_slot = {}
        node_first_time = {}
        batch_max_time = 0
        for i in range(self.batch_size):
            user_index = idx*self.batch_size+i
            if user_index >= len(self.user_permutation):
                break
            batch_max_time = self.user_seq_times[user_index][1]
            i_user = self.user_permutation[user_index]  # [0, 1, ..., 199]
            self.active_users[i] = i_user
            j = self.all_user_seq[i_user]  # 0
            max_j = self.sequences_count[i_user]  # 用户i所拥有的序列数目
            # if self.usage == Usage.MIN_SEQ_LENGTH:
            #     max_j = self.min_seq_count
            # if self.usage == Usage.CUSTOM:
            #     max_j = min(max_j, self.custom_seq_count)  # use either the users maxima count or limit by custom count
            # if j >= max_j:  # 用户i的所有子序列都已经被使用过了！
            #     # replace this user in current sequence:
            #     i_user = self.user_permutation[self.next_user_idx]  # 取第201个用户作为下一个用户
            #     j = 0
            #     self.active_users[i] = i_user
            #     self.active_user_seq[i] = j
            #     self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
            #     while self.user_permutation[self.next_user_idx] in self.active_users:  # 循环查找不在活跃用户里的新用户
            #         self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
            #     # TODO: throw exception if wrapped around!
            # use this user:
            reset_h.append(j == 0)  # 添加新用户时,要为他重置h  True or False
            seqs.append(torch.tensor(self.sequences[i_user][j]))
            times.append(torch.tensor(self.sequences_times[i_user][j]))
            time_slots.append(torch.tensor(self.sequences_time_slots[i_user][j]))
            time_hours.append(torch.tensor(self.sequences_hours[i_user][j]))
            coords.append(torch.tensor(self.sequences_coords[i_user][j]))
            categorys.append(torch.tensor(self.sequences_categorys[i_user][j]))

            lbls.append(torch.tensor(self.sequences_labels[i_user][j]))
            lbl_times.append(torch.tensor(self.sequences_lbl_times[i_user][j]))
            lbl_time_slots.append(torch.tensor(self.sequences_lbl_time_slots[i_user][j]))
            lbl_coords.append(torch.tensor(self.sequences_lbl_coords[i_user][j]))
            for k in range(20):
                src_list.append(int(i_user))
                dst_list.append(int(self.sequences[i_user][j][k])+len(self.users))
                ts_list.append(self.sequences_times[i_user][j][k])
                # feat_list.append(self.edge_feats[0])
                feat_type.append(0)
                # weekday_list.append(self.sequences_weekdays[i_user][j][k])
                # month_list.append(self.sequences_months[i_user][j][k])
                # hour_list.append(self.sequences_hours[i_user][j][k])
                # dst_list.append(int(i_user))
                # src_list.append(int(self.sequences[i_user][j][k])+len(self.users))
                # ts_list.append(self.sequences_times[i_user][j][k])
                # feat_list.append(self.edge_feats[0])
                # feat_type.append(0)
                # weekday_list.append(self.sequences_weekdays[i_user][j][k])
                # month_list.append(self.sequences_months[i_user][j][k])
                # hour_list.append(self.sequences_hours[i_user][j][k])
                # print("node_id",int(self.sequences[i_user][j][k])+len(self.users),i,k)
                if int(self.sequences[i_user][j][k])+len(self.users) in node_index_key:
                    # node_index_key[int(self.sequences[i_user][j][k])+len(self.users)] = i*20+k
                    # node_index_key[int(self.sequences[i_user][j][k])+len(self.users)] = -1
                    node_repeat_array[int(self.sequences[i_user][j][k])+len(self.users)].append(i*20+k)
                    node_repeat_time[int(self.sequences[i_user][j][k])+len(self.users)].append(self.sequences_times[i_user][j][k])
                    node_repeat_slot[int(self.sequences[i_user][j][k])+len(self.users)].append([self.sequences_months[i_user][j][k],self.sequences_weekdays[i_user][j][k],self.sequences_hours[i_user][j][k]])
                else:
                    node_index_key[int(self.sequences[i_user][j][k])+len(self.users)] = i*20+k
                    # node_first_nid.append(int(self.sequences[i_user][j][k])+len(self.users))
                    node_first_time[int(self.sequences[i_user][j][k])+len(self.users)] = self.sequences_times[i_user][j][k]
                    node_repeat_array[int(self.sequences[i_user][j][k])+len(self.users)] = []
                    node_repeat_time[int(self.sequences[i_user][j][k])+len(self.users)] = [self.sequences_times[i_user][j][k]]
                    node_repeat_slot[int(self.sequences[i_user][j][k])+len(self.users)] = [[self.sequences_months[i_user][j][k],self.sequences_weekdays[i_user][j][k],self.sequences_hours[i_user][j][k]]]
                # key = "{}_{}".format(self.sequences[i_user][j][k],self.sequences[i_user][j][k+1])
                src_list.append(int(self.sequences[i_user][j][k])+len(self.users))
                dst_list.append(int(self.sequences_labels[i_user][j][k])+len(self.users))
                ts_list.append(self.sequences_times[i_user][j][k])
                # feat_list.append(self.edge_feats[1])
                feat_type.append(1)
                weekday_list.append(self.sequences_weekdays[i_user][j][k])
                month_list.append(self.sequences_months[i_user][j][k])
                hour_list.append(self.sequences_hours[i_user][j][k])
                # dst_list.append(int(self.sequences[i_user][j][k])+len(self.users))
                # src_list.append(int(self.sequences[i_user][j][k+1])+len(self.users))
                # ts_list.append(self.sequences_times[i_user][j][k+1])
                # feat_list.append(self.edge_feats[1])
                # feat_type.append(1)
                # weekday_list.append(self.sequences_weekdays[i_user][j][k+1])
                # month_list.append(self.sequences_months[i_user][j][k+1])
                # hour_list.append(self.sequences_hours[i_user][j][k+1])


            self.all_user_seq[i_user] += 1  # 如果用户i再次被选择为活跃用户时,j加1使得选择i的后续的子序列
        # print(node_repeat_array)
        x = torch.stack(seqs, dim=1)
        t = torch.stack(times, dim=1)
        t_slot = torch.stack(time_slots, dim=1)
        t_hour = torch.stack(time_hours, dim=1)
        s = torch.stack(coords, dim=1)
        cat = torch.stack(categorys, dim=1)

        y = torch.stack(lbls, dim=1)
        y_t = torch.stack(lbl_times, dim=1)
        y_t_slot = torch.stack(lbl_time_slots, dim=1)
        y_s = torch.stack(lbl_coords, dim=1)
        ts_list = torch.tensor(ts_list,dtype=torch.double,requires_grad=False)
        feat_list = torch.from_numpy(np.array(feat_list))
        feat_type_list = torch.from_numpy(np.array(feat_type))
        weekday_list_1 = torch.from_numpy(np.array(weekday_list))
        month_list_1 = torch.from_numpy(np.array(month_list))
        hour_list_1 = torch.from_numpy(np.array(hour_list))
        # print("get normal data end", time.time())
        total_nodes = len(self.users) + self.loc_count


        # start_time = time.time()
        positive_pair_g = dgl.graph((src_list, dst_list),num_nodes=total_nodes)
        positive_pair_g.edata['timestamp'] = ts_list
        positive_pair_g.edata['feats_type'] = feat_type_list
        positive_pair_g.ndata[dgl.NID] = torch.arange(positive_pair_g.num_nodes())
        edge_number = len(src_list)
        # end_time = time.time()
        # print("step1", end_time-start_time)

        # 正负样本
        positive_pair_g = dgl.transforms.compact_graphs(positive_pair_g)
        timestamps = []
        batch_graphs = []
        src_array = np.unique(np.array(src_list))
        # print('src:',src_array)
        dst_array = np.unique(np.array(dst_list))
        # print('dst:',dst_array)
        # 正负样本
        # neg_dst_array = np.unique(np.array(neg_srcdst_raw[1].tolist()))
        # print(neg_dst_array)
        # dst_array = np.append(dst_array,neg_dst_array)
        # 正负样本
        # print(dst_array)
        # nodes_list = list(np.unique(np.append(src_array,dst_array)))
        nodes_list = list(src_array)
        # print(nodes_list)
        # exit()
        # start_time = time.time()
        ts = ts_list[-1]
        min_ts = torch.min(ts_list)

        subg,hour_loc_g = self.sample_blocks(self.g,self.total_g,self.fri_g,self.hour_g,nodes_list,self.days,timestamp=ts,min_ts=min_ts)
        # for j in range(len(subg)):
        #     subg[j].ndata['timestamp'] = ts.repeat(subg[j].num_nodes())
        #     # nodes_id.append(subg[j].srcdata[dgl.NID])
        #     batch_graphs.append(subg[j])
        # first_src,first_dst = subg[0].edges()
        # first_src_array = np.array(first_src.tolist())
        # first_dst_array = np.array(first_dst.tolist())
        # first_node = list(np.unique(np.append(first_src_array,first_dst_array) ) ) 
        # second_subg,_ = self.sample_blocks(self.g,self.total_g,self.fri_g,self.hour_g,first_node,self.days,timestamp=ts,min_ts=min_ts)
        # merge_g = dgl.merge([batch_graphs[0],batch_graphs[1]])
        # merge_g = dgl.merge([subg[0],second_subg[0]])
        # merge_g = second_subg[0]
        merge_g = subg[0]
        src, dst = merge_g.edges()
        merge_src = src.tolist()
        merge_dst = dst.tolist()

        merge_feats_type = merge_g.edata['feats_type'].tolist()
        new_merge_src = []
        new_merge_dst = []
        new_merge_time = []
        new_merge_month = []
        new_merge_weekday = []
        new_merge_hour = []
        new_merge_feats_type = []
        for k in range(len(merge_src)):
            src_node_id = merge_src[k]
            dst_node_id = merge_dst[k]
            new_merge_src.append(src_node_id)
            new_merge_dst.append(dst_node_id)
            if dst_node_id in node_repeat_time:
                new_merge_time.append(node_repeat_time[dst_node_id][0])
                new_merge_month.append(node_repeat_slot[dst_node_id][0][0])
                new_merge_weekday.append(node_repeat_slot[dst_node_id][0][1])
                new_merge_hour.append(node_repeat_slot[dst_node_id][0][2])
            else:
                new_merge_time.append(0)
                new_merge_month.append(12)
                new_merge_weekday.append(7)
                new_merge_hour.append(24)
            new_merge_feats_type.append(merge_feats_type[k])
            if dst_node_id in node_repeat_array:
                for repeat_index in range(len(node_repeat_array[dst_node_id])):
                    sq_index = node_repeat_array[dst_node_id][repeat_index]
                    new_merge_src.append(src_node_id)
                    new_merge_dst.append(sq_index*total_nodes+dst_node_id)
                    new_merge_time.append(node_repeat_time[dst_node_id][1+repeat_index])
                    new_merge_month.append(node_repeat_slot[dst_node_id][1+repeat_index][0])
                    new_merge_weekday.append(node_repeat_slot[dst_node_id][1+repeat_index][1])
                    new_merge_hour.append(node_repeat_slot[dst_node_id][1+repeat_index][2])
                    new_merge_feats_type.append(merge_feats_type[k])


        new_merge_g = dgl.graph((new_merge_src, new_merge_dst))
        new_merge_g.edata['timestamp'] = torch.tensor(new_merge_time,dtype=torch.double,requires_grad=False)
        new_merge_g.edata['month'] = torch.tensor(new_merge_month,dtype=torch.int64,requires_grad=False)
        new_merge_g.edata['weekday'] = torch.tensor(new_merge_weekday,dtype=torch.int64,requires_grad=False)
        new_merge_g.edata['hour'] = torch.tensor(new_merge_hour,dtype=torch.int64,requires_grad=False)
        new_merge_g.edata['feats_type'] = torch.from_numpy(np.array(new_merge_feats_type))
        new_merge_g.ndata[dgl.NID] = torch.arange(new_merge_g.num_nodes())
        new_merge_g.ndata['timestamp'] = torch.from_numpy(np.zeros(new_merge_g.num_nodes(),dtype=np.double ))
        new_g = dgl.transforms.compact_graphs(new_merge_g)
        new_g.ndata['node_index'] = new_g.ndata[dgl.NID]
        new_g.ndata[dgl.NID] = new_g.ndata[dgl.NID] % total_nodes


        # merge_nid = merge_g.ndata[dgl.NID]
        # tail_nid = merge_nid[dst]
        # first_time = [node_first_time[int(n)] for n in tail_nid]
        # merge_g.edata['timestamp'] = torch.tensor(first_time,dtype=torch.double,requires_grad=False)
        # merge_g.ndata['node_index'] = merge_g.ndata[dgl.NID]
        # def sample_subgraph(subg, seeds):
        #     blocks = []
        #     count = 0
        #     for i in range(len(seeds)):
        #         pair = seeds[i]
        #         if node_index_key[pair] < i :
        #             count+=1
        #             # print(subg.ndata[dgl.NID])
        #             block = dgl.in_subgraph(subg, [pair])
        #             # print(block.ndata[dgl.NID])
        #             block.ndata['node_index'] = torch.from_numpy(np.ones(block.num_nodes(),dtype=np.int64 ) * i * total_nodes)+block.ndata[dgl.NID]
        #             block.edata['timestamp'] = ts_list[i].repeat(block.num_edges())
        #             blocks.append(dgl.transforms.compact_graphs(block))
        #         # if i > 1 :
        #         #     break
        #     blocks.append(dgl.transforms.compact_graphs(subg))
        #     # print("count:",count)
        #     return dgl.batch(blocks)
        # new_g = sample_subgraph(merge_g, src_list)
        # print(sampled_subgraphs.ndata[dgl.NID])
        # exit()

        self.hour_loc_g = hour_loc_g
        # for i, edge in enumerate(zip(src_list, dst_list)):
        #     ts = ts_list[i]
        #     timestamps.append(ts)
        #     subg = self.sample_blocks(self.g,self.loc_g,self.fri_g,list(edge),timestamp=ts)
        #     for j in range(len(subg)):
        #         subg[j].ndata['timestamp'] = ts.repeat(subg[j].num_nodes())
        #         # nodes_id.append(subg[j].srcdata[dgl.NID])
        #         batch_graphs.append(subg[j])

            # break
        # blocks = [dgl.batch(batch_graphs)]
        # print(batch_graphs[0].nodes()[:10])
        # print(batch_graphs[0].ndata[dgl.NID][:10])
        # print(len(batch_graphs[0].ndata[dgl.NID]),len(batch_graphs[1].ndata[dgl.NID]))
        # new_g = dgl.merge([batch_graphs[0],batch_graphs[1]])
        # new_g = dgl.transforms.compact_graphs(new_merge_g)
        # new_g = sampled_subgraphs
        # print(new_g.ndata[dgl.NID])
        # print(new_g.ndata['node_index'])
        # exit()
        # new_g = dgl.transforms.compact_graphs(batch_graphs[0])
        # print(len(new_g.ndata[dgl.NID]))
        # print(len(dgl.batch(batch_graphs).ndata[dgl.NID]))
        blocks = [new_g]
        # print()
        # print("get graph data end", time.time())
        # end_time = time.time()
        # print("step3", end_time-start_time)
        # exit()
        self.node_index_key = node_index_key
        self.positive_pair_g = positive_pair_g
        self.negative_pair_g = positive_pair_g
        # 正负样本
        # self.negative_pair_g = negative_pair_g
        # 正负样本
        self.block = blocks[0]
        # print(self.block.ndata['timestamp'])

        # if self.split == Split.TEST:
        #     temp_ts_list = torch.cat([self.g.edata['timestamp'], ts_list],dim=0)
        #     # print(temp_ts_list)
        #     # temp_feat = torch.cat([self.g.edata['feats'], feat_list],dim=0)
        #     temp_feat_type = torch.cat([self.g.edata['feats_type'], feat_type_list],dim=0)
        #     # temp_weekdays = torch.cat([self.g.edata['weekday'], weekday_list_1],dim=0)
        #     # temp_months = torch.cat([self.g.edata['month'], month_list_1],dim=0)
        #     # temp_hours = torch.cat([self.g.edata['hour'], hour_list_1],dim=0)
        #     self.g = dgl.add_edges(self.g,src_list,dst_list)
        #     self.g.edata['timestamp'] = temp_ts_list
        #     # self.g.edata['feats'] = temp_feat
        #     self.g.edata['feats_type'] = temp_feat_type
        #     # self.g.edata['weekday'] = temp_weekdays
        #     # self.g.edata['month'] = temp_months
        #     # self.g.edata['hour'] = temp_hours
        #     self.g.ndata[dgl.NID] = torch.arange(self.g.num_nodes())
        #     # self.g.edata[dgl.EID] = torch.arange(self.g.num_edges())

        # print("fix graph end", time.time())
        return x, t, t_slot, s, y, y_t, y_t_slot, y_s, reset_h, torch.tensor(self.active_users), t_hour, cat

    def sampler_frontier(self,
                         block_id,
                         g,
                         seed_nodes,
                         timestamp,
                         min_ts,
                         sample_type,
                         days):
        full_neighbor_subgraph = dgl.in_subgraph(g, seed_nodes)
        if sample_type =='loc':
            t = []
            f = []
            f_t = []
            w = []
            m = []
            h = []
            for i in range(len(seed_nodes)):
                t.append(0)
                w.append(7)
                m.append(12)
                h.append(24)
                if seed_nodes[i] >= len(self.users):
                    f.append(self.edge_feats[1])
                    f_t.append(1)
                else:
                    f.append(self.edge_feats[0])
                    f_t.append(0)
            # data = {'timestamp': torch.tensor(t,dtype=torch.double), 'feats': torch.from_numpy(np.array(f)), 'feats_type': torch.from_numpy(np.array(f_t)), 'weekday': torch.from_numpy(np.array(w)), 'month': torch.from_numpy(np.array(m)), 'hour': torch.from_numpy(np.array(h))}
            data = {'timestamp': torch.tensor(t,dtype=torch.double), 'feats_type': torch.from_numpy(np.array(f_t))}
            # print(seed_nodes)
            # print(data)
            full_neighbor_subgraph = dgl.add_edges(full_neighbor_subgraph,seed_nodes, seed_nodes, data=data)

        if sample_type == 'all':
            temporal_subgraph = full_neighbor_subgraph
        else:
            temporal_edge_mask = (full_neighbor_subgraph.edata['timestamp'] < timestamp) * (full_neighbor_subgraph.edata['timestamp'] >= (min_ts - days*86400)) + (
                full_neighbor_subgraph.edata['timestamp'] <= 0)
            temporal_subgraph = dgl.edge_subgraph(
                full_neighbor_subgraph, temporal_edge_mask, relabel_nodes=False)


        # print(temporal_subgraph.edges())
        # Map preserve ID
        temp2origin = temporal_subgraph.ndata[dgl.NID]
        # print(temp2origin)
        # print(temporal_subgraph.nodes())
        # The added new edgge will be preserved hence
        root2sub_dict = dict(
            zip(temp2origin.tolist(), temporal_subgraph.nodes().tolist()))
        temporal_subgraph.ndata[dgl.NID] = g.ndata[dgl.NID][temp2origin]
        try:
            temp_seed_nodes = []
            for n in seed_nodes:
                if int(n) in root2sub_dict:
                    temp_seed_nodes.append(root2sub_dict[int(n)])
            # seed_nodes = [root2sub_dict[int(n)] for n in seed_nodes]
        except Exception as e:
            print(full_neighbor_subgraph.edges())
            print(seed_nodes)
            print(root2sub_dict)
            raise e

        if sample_type == 'top' or sample_type == 'all':
            final_subgraph_temp = self.sampler(g=temporal_subgraph, nodes=temp_seed_nodes)
            unique_edges = set(zip(final_subgraph_temp.edges()[0].tolist(), final_subgraph_temp.edges()[1].tolist()))
            unique_src, unique_dst = zip(*unique_edges)
            final_subgraph = dgl.graph((torch.tensor(unique_src), torch.tensor(unique_dst)),num_nodes=self.g.num_nodes())
            final_subgraph.ndata[dgl.NID] = torch.arange(self.g.num_nodes())
            final_subgraph.edata['feats_type'] = torch.from_numpy(np.ones(final_subgraph.num_edges(),dtype=np.int64 )*1)
            final_subgraph.edata['timestamp'] = torch.from_numpy(np.zeros(final_subgraph.num_edges(),dtype=np.int64 ))
        elif sample_type == 'fri':
            final_subgraph = self.fri_sampler(g=temporal_subgraph, nodes=temp_seed_nodes)
            # print(final_subgraph.edges())
            # exit()
        else:
            final_subgraph = self.loc_sampler(g=temporal_subgraph, nodes=temp_seed_nodes)
        # final_subgraph.remove_self_loop()
        return final_subgraph

        # Temporal Subgraph
    def sample_blocks(self,
                      g,
                      g_loc,
                      g_fri,
                      g_hour,
                      seed_nodes,
                      days,
                      timestamp,
                      min_ts
                      ):
        # print(seed_nodes)
        blocks = []
        nodes = []
        # for n in seed_nodes:
        #     if n < g.num_nodes():
        #         nodes.append(n)

        # if len(seed_nodes)>0:
        #     frontier = self.sampler_frontier(0, g, seed_nodes, timestamp, min_ts, 'fri', days)
        #     # print(frontier.num_edges(), len(seed_nodes))
        #     # print(frontier.ndata)
        #     # seed_nodes = [int(n.item()) for n in seed_nodes]
        #     blocks.append(frontier)

        loc_nodes = []
        fri_nodes = []
        hour_nodes = []
        for n in seed_nodes:
            if n>=len(self.users):
                loc_nodes.append(n)
                hour_nodes.append(n-len(self.users))
            else:
                loc_nodes.append(n)
                fri_nodes.append(n)

        # if len(fri_nodes)>0:
        #     # print(fri_nodes)
        #     frontier_fri = self.sampler_frontier(0, g_fri, fri_nodes, timestamp, min_ts, 'fri',days)
        #     # g_fri_temp = self.org_sampler(g=g_fri, nodes=fri_nodes)
        #     # frontier_fri = dgl.in_subgraph(g_fri, fri_nodes, relabel_nodes=True)
        #     blocks.append(frontier_fri)
        #     # print(frontier_fri.edges())
        #     # print(frontier_fri.ndata)

        if len(loc_nodes)>0:
            # print(loc_nodes)
            frontier_loc = self.sampler_frontier(0, g_loc, loc_nodes, timestamp, min_ts, 'loc',days)
            # frontier_loc = self.org_sampler(g=g_loc, nodes=loc_nodes)

            # frontier_loc = dgl.in_subgraph(g_loc, loc_nodes, relabel_nodes=True)
            # temp2origin = g_loc_temp.ndata[dgl.NID]
            # root2sub_dict = dict(zip(temp2origin.tolist(), temporal_subgraph.nodes().tolist()))
            # temporal_subgraph.ndata[dgl.NID] = g.ndata[dgl.NID][temp2origin]
            # seed_nodes = [root2sub_dict[int(n)] for n in seed_nodes]
            blocks.append(frontier_loc)
            # print(hour_nodes)
            # exit()
            # frontier_hour = self.sampler_frontier(0, g_hour, hour_nodes, timestamp, min_ts, 'all',days)
            # blocks.append(frontier_hour)
            # print(frontier_loc.edges())
            # print(frontier_loc.ndata)



        #block = transform.to_block(frontier,seed_nodes)
        # block = frontier
        # if self.return_eids:
        #     self.assign_block_eids(block, frontier)


        return blocks,frontier_loc
        # return blocks,frontier_hour