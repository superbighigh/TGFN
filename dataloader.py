import os.path
import sys
import dgl
from datetime import datetime
import torch
from dataset import PoiDataset, Usage
import numpy as np
from collections import defaultdict
from scipy.sparse import lil_matrix
import pandas as pd


class PoiDataloader():
    """ Creates datasets from our prepared Gowalla/Foursquare data files.
    The file consist of one check-in per line in the following format (tab separated):

    <user-id> <timestamp> <latitude> <longitude> <location-id>

    Check-ins for the same user have to be on continuous lines.
    Ids for users and locations are recreated and continuous from 0.
    """

    def __init__(self, setting ,max_users=0, min_checkins=0):
        """ max_users limits the amount of users to load.
        min_checkins discards users with less than this amount of checkins.
        """

        self.max_users = max_users  # 0
        self.min_checkins = min_checkins  # 101

        self.user2id = {}
        self.poi2id = {}
        self.poi2gps = {}  # 自己加的

        self.users = []
        self.times = []  # 二重列表,每个元素是active user所对应POIs的访问timestamp,按时间顺序排序
        self.time_slots = []
        self.coords = []  # 二重列表,每个元素是active user所对应POIs的GPS坐标,按时间顺序排序
        self.locs = []  # 二重列表,每个元素是active user所对应POIs,按时间顺序排序
        self.weekdays = []
        self.months = []
        self.hours = []
        self.categorys = []
        self.e_feat = torch.randn(4, 10).numpy()
        self.cat_count = 0
        self.user_checkins = defaultdict(list)
        self.setting = setting

    def create_dataset(self, sequence_length, batch_size, split, days, fri_sampler_count, loc_sampler_count, usage=Usage.MAX_SEQ_LENGTH, custom_seq_count=1):
        return PoiDataset(self.users.copy(),
                          self.times.copy(),
                          self.time_slots.copy(),
                          self.coords.copy(),
                          self.locs.copy(),
                          sequence_length,
                          batch_size,
                          split,
                          usage,
                          len(self.poi2id),
                          custom_seq_count,
                          self.loc_g,
                          self.fri_g,
                          self.cat_g,
                          days,
                          fri_sampler_count,
                          loc_sampler_count,
                          self.weekdays.copy(),
                          self.months.copy(),
                          self.hours.copy(),
                          self.hour_g,
                          self.categorys.copy(),
                          self.setting)

    def user_count(self):
        return len(self.users)

    def locations(self):
        return len(self.poi2id)

    def checkins_count(self):
        count = 0
        for loc in self.locs:
            count += len(loc)
        return count

    def read(self, file):
        if not os.path.isfile(file):
            print('[Error]: Dataset not available: {}. Please follow instructions under ./data/README.md'.format(file))
            sys.exit(1)
        pre_suffix = file.split('./data/')[1].split('_')[0]
        # collect all users with min checkins:
        self.read_users(file)
        # collect checkins for all collected users:
        self.read_pois(file,pre_suffix)
        self.load_loc_graph('data/{}_poi_spatial_triplets.txt'.format(pre_suffix))
        self.load_fri_graph('data/gowalla_friend.txt')
        # self.load_cat_graph('data/gowalla_poi_category_relation.txt')
        self.load_cat_graph()

    def read_users(self, file):
        f = open(file, 'r')
        lines = f.readlines()

        prev_user = int(lines[0].split('\t')[0])
        visit_cnt = 0
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if user == prev_user:
                visit_cnt += 1
            else:
                if visit_cnt >= self.min_checkins:
                    self.user2id[prev_user] = len(self.user2id)
                # else:
                #    print('discard user {}: to few checkins ({})'.format(prev_user, visit_cnt))
                prev_user = user
                visit_cnt = 1
                if 0 < self.max_users <= len(self.user2id):
                    break  # restrict to max users

    def read_pois(self, file, pre_suffix):
        f = open(file, 'r')
        lines = f.readlines()

        # store location ids
        user_time = []
        user_coord = []
        user_loc = []
        user_time_slot = []
        user_weekday = []
        user_month = []
        user_hour = []
        user_category = []
        

        category_dict = {}
        
        loc_hour_json = {}

        prev_user = int(lines[0].split('\t')[0])
        prev_user = self.user2id.get(prev_user)  # from 0
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if self.user2id.get(user) is None:
                continue  # user is not of interest(inactive user)
            user = self.user2id.get(user)  # from 0

            if pre_suffix == 'gowalla':
                time = (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ") - datetime(2009, 2, 4)).total_seconds()  # unix seconds
            elif pre_suffix == 'foursquare':
                time = (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ") - datetime(2012, 4, 1)).total_seconds()  # unix seconds
            else:
                print('[Error]: Dataset not available: {}. Please follow instructions under ./data/README.md'.format(pre_suffix))
                sys.exit(1)
            # 自己加的time slot, 将一周的时间分成24 * 7个时间槽
            time_slot = (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ")).weekday() * 24 + (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ")).hour
            lat = float(tokens[2])
            long = float(tokens[3])
            coord = (lat, long)
            weekday = (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ")).weekday()
            month = (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ")).month - 1
            h = (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ")).hour
            minute = (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ")).minute
            hour = int(h / 4)
            # h = h*2 + int(minute/30)
            h = int(h/2)
            # if weekday<5:
            #     weekday = 0
            # else:
            #     weekday = 1
            
            # if month>=2 and month<=4:
            #     month = 0
            # elif month>=5 and month<=7:
            #     month = 1
            # elif month>=8 and month<=10:
            #     month = 2
            # else:
            #     month = 3
            cat_name = "no"
            if len(tokens) >= 6:
                cat_name = tokens[5]
            
            if cat_name not in category_dict:
                category_dict[cat_name] = self.cat_count
                self.cat_count += 1
            category_id = category_dict[cat_name]
            self.user_checkins[user].insert(0,(time, category_id))

            location = int(tokens[4])  # location nr
            if self.poi2id.get(location) is None:  # get-or-set locations
                self.poi2id[location] = len(self.poi2id)
                self.poi2gps[self.poi2id[location]] = coord
            location = self.poi2id.get(location)  # from 0
            # if location == 63:
            #     print(coord)
            #     exit()
            if h in loc_hour_json:
                if location in loc_hour_json[h]:
                    loc_hour_json[h][location]+=1
                else:
                    loc_hour_json[h][location]=1
            else:
                loc_hour_json[h] = {location:1}

            if user == prev_user:
                # Because the check-ins for every user is sorted in descending chronological order in the file
                user_time.insert(0, time)  # insert in front!
                user_time_slot.insert(0, time_slot)
                user_coord.insert(0, coord)
                user_loc.insert(0, location)
                user_weekday.insert(0,weekday)
                user_month.insert(0,month)
                user_hour.insert(0,hour)
                user_category.insert(0,category_id)
            else:
                self.users.append(prev_user)  # 添加用户
                self.times.append(user_time)  # 添加列表
                self.time_slots.append(user_time_slot)
                self.coords.append(user_coord)
                self.locs.append(user_loc)
                self.weekdays.append(user_weekday)
                self.months.append(user_month)
                self.hours.append(user_hour)
                self.categorys.append(user_category)
                # print(len(user_time) == len(user_time_slot) == len(user_loc) == len(user_coord))
                # restart:
                prev_user = user
                user_time = [time]
                user_time_slot = [time_slot]
                user_coord = [coord]
                user_loc = [location]
                user_weekday = [weekday]
                user_month = [month]
                user_hour = [hour]
                user_category = [category_id]

        # process also the latest user in the for loop
        self.users.append(prev_user)
        self.times.append(user_time)
        self.time_slots.append(user_time_slot)
        self.coords.append(user_coord)
        self.locs.append(user_loc)
        self.weekdays.append(user_weekday)
        self.months.append(user_month)
        self.hours.append(user_hour)
        self.categorys.append(user_category)
        # print(category_dict)
        # exit()

        src = []
        dst = []
        num = []
        t = []
        for h in loc_hour_json:
            for loc in loc_hour_json[h]:
                src.append(h)
                dst.append(loc)
                num.append(loc_hour_json[h][loc])
                t.append(0)
        self.hour_g = dgl.graph((np.array(src), np.array(dst)),num_nodes=len(self.poi2id))
        self.hour_g.edata['num'] = torch.from_numpy(np.array(num))
        self.hour_g.ndata[dgl.NID] = torch.arange(self.hour_g.num_nodes())
        self.hour_g.edata[dgl.EID] = torch.arange(self.hour_g.num_edges())
        self.hour_g.edata['timestamp'] = torch.tensor(t,dtype=torch.double,requires_grad=False)

    def load_loc_graph(self, loc_path):
        u_list, i_list, ts_list, feat_loc, feat_type, weekdays, months, hours = [],[],[],[],[],[],[],[]
        with open(loc_path) as f:
            for idx, line in enumerate(f):
                e = line.strip().split('\t')
                u = int(e[0])
                i = int(e[1])
                u_list.append(u)
                i_list.append(i)
                ts_list.append(0)
                feat_loc.append(self.e_feat[2])
                feat_type.append(2)
                weekdays.append(7)
                months.append(12)
                hours.append(24)
        src = np.array(u_list)
        dst = np.array(i_list)
        self.loc_g = dgl.graph((src, dst),num_nodes=len(self.poi2id)+len(self.users))
        self.loc_g.edata['timestamp'] = torch.tensor(ts_list,dtype=torch.double,requires_grad=False)
        # self.loc_g.edata['feats'] = torch.from_numpy(np.array(feat_loc)).float()
        self.loc_g.edata['feats_type'] = torch.from_numpy(np.array(feat_type))
        # self.loc_g.edata['weekday'] = torch.from_numpy(np.array(weekdays))
        # self.loc_g.edata['month'] = torch.from_numpy(np.array(months))
        # self.loc_g.edata['hour'] = torch.from_numpy(np.array(hours))
        self.loc_g.ndata[dgl.NID] = torch.arange(self.loc_g.num_nodes())
        self.loc_g.edata[dgl.EID] = torch.arange(self.loc_g.num_edges())
        # print(self.loc_g.num_nodes())
        # print(self.loc_g.nodes())

    def load_cat_graph(self):
        sequence_counts = self.calculate_category_sequences()

        # 计算类别总数
        categories = sorted(sequence_counts.keys())
        category_count = len(categories)

        # 计算转移概率并生成稀疏矩阵
        self.cat_g, self.cat_relation_graph = self.calculate_transition_probabilities(sequence_counts, category_count)

    def calculate_category_sequences(self):
        """
        统计地点类别之间的顺序关系出现次数。
        """
        sequence_counts = defaultdict(lambda: defaultdict(int))

        for user, checkins in self.user_checkins.items():
            for i in range(len(checkins) - 1):
                category_a = checkins[i][1]
                category_b = checkins[i + 1][1]
                sequence_counts[category_a][category_b] += 1

        return sequence_counts

    def calculate_transition_probabilities(self, sequence_counts, category_count):
        """
        计算每种来源类别的转移概率，并生成稀疏矩阵。
        """
        # categories = sorted(sequence_counts.keys())
        # category_to_index = {category: idx for idx, category in enumerate(categories)}

        transition_graph = lil_matrix((category_count, category_count), dtype=np.float32)

        src_list = []
        dst_list = []
        for category_a, transitions in sequence_counts.items():
            # 获取转移次数最多的20个目标类别
            top_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:self.setting.cat_relation_count]

            # 计算总转移次数
            total_transitions = sum(count for _, count in top_transitions)

            # 计算转移概率并填入稀疏矩阵
            for category_b, count in top_transitions:
                prob = count / total_transitions
                # idx_a = category_to_index[category_a]
                # idx_b = category_to_index[category_b]
                transition_graph[category_a, category_b] = prob
                src_list.append(category_a)
                src_list.append(category_b)
                dst_list.append(category_b)
                dst_list.append(category_a)

        src = np.array(src_list)
        dst = np.array(dst_list)
        cat_relation_graph = dgl.graph((src, dst),num_nodes=category_count)
        # 打印抽查的转移概率结果
        # print("Sample transition probabilities:")
        # for category_a, transitions in sequence_counts.items():
        #     top_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:20]
        #     for category_b, count in top_transitions:
        #         prob = count / sum(count for _, count in top_transitions)
        #         print(f"{category_a} -> {category_b}: {prob:.4f}")
        #     break  # 只打印第一个来源类别的结果

        return transition_graph,cat_relation_graph

    # def load_cat_graph(self, loc_path):
    #     u_list, i_list, ts_list, feat_loc, feat_type, weekdays, months, hours = [],[],[],[],[],[],[],[]
    #     with open(loc_path) as f:
    #         for idx, line in enumerate(f):
    #             e = line.strip().split('\t')
    #             u = int(e[0])
    #             i = int(e[1])
    #             u_list.append(u)
    #             i_list.append(i)
    #             u_list.append(i)
    #             i_list.append(u)
    #             ts_list.append(0)
    #             ts_list.append(0)
    #             # feat_loc.append(self.e_feat[4])
    #             feat_type.append(4)
    #             feat_type.append(4)
    #             # weekdays.append(7)
    #             # months.append(12)
    #             # hours.append(24)
    #     src = np.array(u_list)
    #     dst = np.array(i_list)
    #     self.cat_g = dgl.graph((src, dst),num_nodes=len(self.poi2id)+len(self.users))
    #     self.cat_g.edata['timestamp'] = torch.tensor(ts_list,dtype=torch.double,requires_grad=False)
    #     # self.loc_g.edata['feats'] = torch.from_numpy(np.array(feat_loc)).float()
    #     self.cat_g.edata['feats_type'] = torch.from_numpy(np.array(feat_type))
    #     # self.loc_g.edata['weekday'] = torch.from_numpy(np.array(weekdays))
    #     # self.loc_g.edata['month'] = torch.from_numpy(np.array(months))
    #     # self.loc_g.edata['hour'] = torch.from_numpy(np.array(hours))
    #     self.cat_g.ndata[dgl.NID] = torch.arange(self.cat_g.num_nodes())
    #     self.cat_g.edata[dgl.EID] = torch.arange(self.cat_g.num_edges())
    #     # print(self.loc_g.num_nodes())
    #     # print(self.loc_g.nodes())

    def load_fri_graph(self, fri_path):
        u_list, i_list, ts_list, feat_fri, feat_type, weekdays, months, hours = [],[],[],[],[],[],[],[]
        with open(fri_path) as f:
            for idx, line in enumerate(f):
                e = line.strip().split('\t')
                u = int(e[0])
                i = int(e[1])
                if self.user2id.get(u) and self.user2id.get(i):  # only focus on active users
                    user_id1 = int(self.user2id.get(u))
                    user_id2 = int(self.user2id.get(i))
                    u_list.append(user_id1)
                    u_list.append(user_id2)
                    i_list.append(user_id2)
                    i_list.append(user_id1)
                    ts_list.append(0)
                    ts_list.append(0)
                    # idx_list.append(idx*2)
                    # idx_list.append(idx*2+1)
                    feat_fri.append(self.e_feat[3])
                    feat_fri.append(self.e_feat[3])
                    feat_type.append(3)
                    feat_type.append(3)
                    weekdays.append(7)
                    weekdays.append(7)
                    months.append(12)
                    months.append(12)
                    hours.append(24)
                    hours.append(24)
        src = np.array(u_list)
        dst = np.array(i_list)
        self.fri_g = dgl.graph((src, dst),num_nodes=len(self.users))
        self.fri_g.edata['timestamp'] = torch.tensor(ts_list,dtype=torch.double,requires_grad=False)
        # self.fri_g.edata['feats'] = torch.from_numpy(np.array(feat_fri)).float()
        self.fri_g.edata['feats_type'] = torch.from_numpy(np.array(feat_type))
        # self.fri_g.edata['weekday'] = torch.from_numpy(np.array(weekdays))
        # self.fri_g.edata['month'] = torch.from_numpy(np.array(months))
        # self.fri_g.edata['hour'] = torch.from_numpy(np.array(hours))
        self.fri_g.ndata[dgl.NID] = torch.arange(self.fri_g.num_nodes())
        # self.fri_g.edata[dgl.EID] = torch.arange(self.fri_g.num_edges())