# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import os
import time
import utils
from utils.parser import get_parser, parse_gpus, BaseConfig
import genotypes.genotypes as gt


class SearchStageConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Search Dag config")
        parser.add_argument('--type', default="KD", help='candidate=[KD, ArchHINT, SearchEval, SearchEvalCurriculum]')
        # ================= file settings ==================
        parser.add_argument('--name', required=True)
        parser.add_argument('--save', type=str, default='EXP', help='experiment name')
        # ================= dataset settings ==================
        parser.add_argument('--dataset', type=str, default='cifar10', help='CIFAR10')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
        parser.add_argument('--cutout_length', type=int, default=0, help='cutout length')
        parser.add_argument('--advanced', type=int, default=0, help='advanced data transform. apply resize (224,224)')
        # ================= optimizer settings ==================
        parser.add_argument('--w_lr', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--w_lr_min', type=float, default=0.001, help='minimum lr for weights')
        parser.add_argument('--w_momentum', type=float, default=0.9, help='momentum for weights')
        parser.add_argument('--w_weight_decay', type=float, default=3e-4,
                            help='weight decay for weights')
        parser.add_argument('--w_grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--alpha_lr', type=float, default=3e-4, help='lr for alpha')
        parser.add_argument('--alpha_weight_decay', type=float, default=1e-3,
                            help='weight decay for alpha')
        # ================= training settings ==================
        parser.add_argument('--epochs', type=int, default=50, help='# of training epochs')
        parser.add_argument('--hint_epochs', nargs="*", type=int, default=[16, 32], help='# of training epochs')
        parser.add_argument('--eval_epochs', type=int, default=100, help='# of training epochs')
        parser.add_argument('--curriculum_epochs', nargs="*", type=int, default=[40, 10], help='# of each curriculum epochs')
        parser.add_argument('--T', type=float, default=10, help='temperature of softmax with temperature')
        parser.add_argument('--l', type=float, default=0.0001, help='ratio between soft target loss and hard target loss')
        parser.add_argument('--g', type=float, default=0.0001, help='Constraint weights for number of cells')
        parser.add_argument('--final_l', type=float, default=None, help='ratio between soft target loss and hard target loss')
        parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--nonkd', type=int, default=0, help='execute KD learning')
        parser.add_argument('--depth_coef', type=float, default=1.0, help='coefficient of depth loss for architecture loss')
        parser.add_argument('--slide_window', type=int, default=3, help='sliding window size')
        parser.add_argument('--discrete', type=int, default=0, help='Use stage specified cell architecture at each stage')
        parser.add_argument('--reset', type=int, default=0, help='Reset network parameters when searching is finished.')
        parser.add_argument('--arch_criterion', type=str, default='l2', help='Constraint Criteria for Parameter Beta. [l1, length, alphal1]')
        parser.add_argument('--stage_macs', nargs="*", type=float, default=[0.33, 0.33, 0.33], help='MACs or percentage of MACs at each stage')
        # ================= model settings ==================
        parser.add_argument('--init_channels', type=int, default=16)
        parser.add_argument('--share_stage', type=int, default=0, help='Search shared stage architecture at each stage')
        parser.add_argument('--spec_cell', type=int, default=0, help='Use stage specified cell architecture at each stage')
        parser.add_argument('--layers', type=int, default=20, help='# of layers')  # 20 layers
        parser.add_argument('--genotype', required=True, help='Cell genotype')
        parser.add_argument('--resume_path', type=str, default=None)
        parser.add_argument('--checkpoint_reset', action='store_true', help='reset resumed model to be as epoch 0')
        parser.add_argument('--teacher_name', type=str, default='densenet121', help='teacher model name')
        parser.add_argument('--teacher_path', type=str, default=None)
        parser.add_argument('--pcdarts', type=int, default=0, help='set PCDARTS model')
        parser.add_argument('--cascade', type=int, default=0, help='set full cascade model(no sliding window)')
        # ================= details ==================
        parser.add_argument('--description', type=str, default='', help='experiment details')
        # ================= others ==================
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        
        return parser
    
    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = '../data/'
        self.genotype = gt.from_str(self.genotype)
        self.gpus = parse_gpus(self.gpus)
        
        self.path = os.path.join(f'results/search_stage_KD/{self.dataset}/', self.name)
        self.exp_name = '{}'.format(args.save)
        self.path = os.path.join(self.path, self.exp_name)
        self.DAG_path = os.path.join(self.path, 'DAG')
        self.plot_path = os.path.join(self.path, 'plots')

        utils.create_exp_dir(self.DAG_path)
        utils.create_exp_dir(self.plot_path)

class TestSearchStageConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Test Searched model config")
        # ================= file settings ==================
        parser.add_argument('--save', type=str, default='EXP', help='experiment name')
        parser.add_argument('--type', default="KD", help='candidate=[KD, ArchHINT, SearchEval, SearchEvalCurriculum]')
        # ================= dataset settings ==================
        parser.add_argument('--dataset', type=str, default='cifar10', help='CIFAR10')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--train_portion', type=float, default=1.0, help='portion of training data')
        parser.add_argument('--cutout_length', type=int, default=0, help='cutout length')
        parser.add_argument('--advanced', type=int, default=0, help='advanced data transform. apply resize (224,224)')
        # ================= model settings ==================
        parser.add_argument('--init_channels', type=int, default=16)
        parser.add_argument('--share_stage', type=int, default=0, help='Search shared stage architecture at each stage')
        parser.add_argument('--spec_cell', type=int, default=0, help='Use stage specified cell architecture at each stage')
        parser.add_argument('--layers', type=int, default=20, help='# of layers')  # 20 layers
        parser.add_argument('--genotype', required=True, help='Cell genotype')
        parser.add_argument('--DAG', required=True, help='DAG genotype')        
        parser.add_argument('--resume_path', required=True, type=str)
        parser.add_argument('--checkpoint_reset', type=int, default=0, help='reset resumed model to be as epoch 0')
        parser.add_argument('--discrete', type=int, default=0, help='Use stage specified cell architecture at each stage')
        parser.add_argument('--slide_window', type=int, default=3, help='sliding window size')
        # ================= details ==================
        parser.add_argument('--description', type=str, default='', help='experiment details')
        # ================= others ==================
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        
        return parser
    
    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = '../data/'
        self.genotype = gt.from_str(self.genotype)
        self.DAG = gt.from_str(self.DAG)
        self.gpus = parse_gpus(self.gpus)

        directory, _ = os.path.split(args.resume_path)
        directory = directory.rstrip(os.path.sep)
        self.path = os.path.join(directory, "test")
        self.path = '{}/{}-{}-{}'.format(self.path, args.save, "discrete" if self.discrete else "relax", time.strftime("%Y%m%d-%H%M%S"))


class SearchDistributionConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Search Dag config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', type=str, default='cifar10', help='CIFAR10')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--w_lr', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--w_lr_min', type=float, default=0.001, help='minimum lr for weights')
        parser.add_argument('--w_momentum', type=float, default=0.9, help='momentum for weights')
        parser.add_argument('--w_weight_decay', type=float, default=3e-4,
                            help='weight decay for weights')
        parser.add_argument('--w_grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=50, help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=16)
        parser.add_argument('--layers', type=int, default=26, help='# of layers')  # 20 layers
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--alpha_lr', type=float, default=3e-4, help='lr for alpha')
        parser.add_argument('--alpha_weight_decay', type=float, default=1e-3,
                            help='weight decay for alpha')

        parser.add_argument('--genotype', required=True, help='Cell genotype')
        parser.add_argument('--local_rank', default=0)
        parser.add_argument('--resume_path', type=str, default=None)

        return parser
    
    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = '../data/'
        self.path = os.path.join('results/search_Stage/cifar_distribution/', self.name)
        self.genotype = gt.from_str(self.genotype)
        self.DAG_path = os.path.join(self.path, 'DAG')
        self.gpus = parse_gpus(self.gpus)
