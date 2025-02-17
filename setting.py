import torch
import argparse
import sys

from network import RnnFactory


class Setting:
    """ Defines all settings in a single place using a command line interface.
    """

    def parse(self):
        self.guess_foursquare = any(['4sq' in argv for argv in sys.argv])  # foursquare has different default args.

        parser = argparse.ArgumentParser()
        if self.guess_foursquare:
            self.parse_foursquare(parser)
        else:
            self.parse_gowalla(parser)
        self.parse_arguments(parser)
        args = parser.parse_args()

        ###### settings ######
        # training
        self.gpu = args.gpu
        self.hidden_dim = args.hidden_dim  # 10
        self.weight_decay = args.weight_decay  # 0.0
        self.learning_rate = args.lr  # 0.01
        self.epochs = args.epochs  # 100
        self.rnn_factory = RnnFactory(args.rnn)  # RNN:0, GRU:1, LSTM:2
        self.is_lstm = self.rnn_factory.is_lstm()  # True or False
        self.lambda_t = args.lambda_t  # 0.01
        self.lambda_s = args.lambda_s  # 100 or 1000

        # data management
        self.dataset_file = './data/{}'.format(args.dataset)
        self.friend_file = './data/{}'.format(args.friendship)
        self.max_users = 0  # 0 = use all available users
        self.sequence_length = 20  # 将用户的所有check-in轨迹划分成固定长度为20的多个子轨迹
        self.batch_size = args.batch_size
        self.min_checkins = 101

        # evaluation        
        self.validate_epoch = args.validate_epoch  # 每5轮验证一次
        self.report_user = args.report_user  # -1

        # log
        self.log_file = args.log_file

        self.trans_loc_file = args.trans_loc_file  # 时间POI graph
        self.trans_loc_spatial_file = args.trans_loc_spatial_file  # 空间POI graph
        self.trans_user_file = args.trans_user_file
        self.trans_interact_file = args.trans_interact_file

        self.lambda_user = args.lambda_user
        self.lambda_loc = args.lambda_loc

        self.use_weight = args.use_weight
        self.use_graph_user = args.use_graph_user
        self.use_spatial_graph = args.use_spatial_graph
        self.sampler_days = args.sampler_days
        self.fri_sampler_count = args.fri_sampler_count
        self.loc_sampler_count = args.loc_sampler_count
        self.sq_count = args.sq_count
        self.sp_count = args.sp_count
        self.score_type = args.score_type
        self.dist_t = args.dist_t
        self.score_relation_type = args.score_relation_type
        self.wj_type = args.wj_type
        self.cat_emb_type = args.cat_emb_type
        self.cat_score_type = args.cat_score_type
        self.scheduler_type = args.scheduler_type
        self.model_name = args.model_name
        self.cat_relation_count = args.cat_relation_count
        self.gcn = args.gcn
        self.relation_type_select = args.relation_type_select
        self.test_model = args.test_model

        ### CUDA Setup ###
        self.device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda', args.gpu)

    def parse_arguments(self, parser):
        # training
        parser.add_argument('--gpu', default=0, type=int, help='the gpu to use')  # -1
        parser.add_argument('--hidden-dim', default=10, type=int, help='hidden dimensions to use')  # 10
        parser.add_argument('--weight_decay', default=0, type=float, help='weight decay regularization')
        parser.add_argument('--lr', default=0.01, type=float, help='learning rate')  # 0.01
        parser.add_argument('--epochs', default=100, type=int, help='amount of epochs')  # 100
        parser.add_argument('--rnn', default='rnn', type=str, help='the GRU implementation to use: [rnn|gru|lstm]')

        # data management
        parser.add_argument('--dataset', default='gowalla_checkins_with_category.txt', type=str,
                            help='the dataset under ./data/<dataset.txt> to load')
        parser.add_argument('--friendship', default='gowalla_friend.txt', type=str,
                            help='the friendship file under ../data/<edges.txt> to load')
        # evaluation        
        parser.add_argument('--validate-epoch', default=5, type=int,
                            help='run each validation after this amount of epochs')
        parser.add_argument('--report-user', default=-1, type=int,
                            help='report every x user on evaluation (-1: ignore)')

        # log
        parser.add_argument('--log_file', default='./results/log_gowalla', type=str,
                            help='存储结果日志')
        parser.add_argument('--trans_loc_file', default='./KGE/POI_graph/gowalla_scheme2_transe_loc_temporal_100.pkl', type=str,
                            help='使用transe方法构造的时间POI转换图')
        parser.add_argument('--trans_user_file', default='', type=str,
                            help='使用transe方法构造的user转换图')
        parser.add_argument('--trans_loc_spatial_file', default='', type=str,
                            help='使用transe方法构造的空间POI转换图')
        parser.add_argument('--trans_interact_file', default='./KGE/POI_graph/gowalla_scheme2_transe_user-loc_100.pkl', type=str,
                            help='使用transe方法构造的用户-POI交互图')
        parser.add_argument('--use_weight', default=False, type=bool, help='应用于GCN的AXW中是否使用W')
        parser.add_argument('--use_graph_user', default=False, type=bool, help='是否使用user graph')
        parser.add_argument('--use_spatial_graph', default=False, type=bool, help='是否使用空间POI graph')
        parser.add_argument('--sampler_days', default=7, type=int, help='采样天数')
        parser.add_argument('--fri_sampler_count', default=-1, type=int, help='采样邻居节点数')
        parser.add_argument('--loc_sampler_count', default=-1, type=int, help='采样邻居节点数')
        parser.add_argument('--sq_count', default=-1, type=int, help='sq关系数量')
        parser.add_argument('--sp_count', default=-1, type=int, help='sp关系数量')
        parser.add_argument('--score_type', default='time-transh', type=str, help='score type')
        parser.add_argument('--dist_t', default='timestamp', type=str, help='dist_t')
        parser.add_argument('--score_relation_type', default='time2v', type=str, help='score relation type')
        parser.add_argument('--wj_type', default='mul', type=str, help='wj_type')
        parser.add_argument('--cat_emb_type', default='transition', type=str, help='cat_emb_type')
        parser.add_argument('--cat_score_type', default='transe', type=str, help='cat_score type')
        parser.add_argument('--scheduler_type', default='normal', type=str, help='scheduler_type')
        parser.add_argument('--model_name', default='no', type=str, help='model_name')
        parser.add_argument('--cat_relation_count', default=20, type=int, help='category关系数量')
        parser.add_argument('--gcn', default=True, type=bool, help='gcn')
        parser.add_argument('--relation_type_select', default="all", type=str, help='relation_type_select')
        parser.add_argument('--test_model', default="no", type=str, help='test_model')

    def parse_gowalla(self, parser):
        # defaults for gowalla dataset
        parser.add_argument('--batch-size', default=100, type=int,  # 200
                            help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=1000, type=float, help='decay factor for spatial data')
        parser.add_argument('--lambda_loc', default=1.0, type=float, help='weight factor for transition graph')
        parser.add_argument('--lambda_user', default=1.0, type=float, help='weight factor for user graph')

    def parse_foursquare(self, parser):
        # defaults for foursquare dataset
        parser.add_argument('--batch-size', default=512, type=int,
                            help='amount of users to process in one pass (batching)')  # 1024
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')
        parser.add_argument('--lambda_loc', default=1.0, type=float, help='weight factor for transition graph')
        parser.add_argument('--lambda_user', default=1.0, type=float, help='weight factor for user graph')

    def __str__(self):
        return (
                   'parse with foursquare default settings' if self.guess_foursquare else 'parse with gowalla default settings') + '\n' \
               + 'use device: {}'.format(self.device)
