# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import os
import time
from utils.parser import get_parser, parse_gpus, BaseConfig
import genotypes.genotypes as gt


class EvaluateStageConfig(BaseConfig):
    def build_parser(self):
        # ======== cifar10 ============
        parser = get_parser("Evaluate searched stagtes of H-DAS+KD config")
        # ================= file settings ==================
        parser.add_argument('--name', required=True)
        parser.add_argument('--save', type=str, default='EXP', help='experiment name')
        # ================= model settings ==================
        parser.add_argument('--genotype', required=True, help='Cell genotype')
        parser.add_argument('--DAG', required=True, help='DAG genotype')
        parser.add_argument('--init_channels', type=int, default=32)
        parser.add_argument('--layers', type=int, default=20, help='# of layers')
        parser.add_argument('--spec_cell', action='store_true', help='Use stage specified cell architecture at each stage')
        parser.add_argument('--resume_path', type=str, default=None)
        parser.add_argument('--teacher_name', type=str, default='densenet121', help='teacher model name')
        parser.add_argument('--teacher_path', type=str, default=None)
        parser.add_argument('--aux_weight', type=float, default=0.4, help='auxiliary loss weight')
        parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path prob')
        # ================= optimizer settings ==================
        parser.add_argument('--lr', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--lr_min', type=float, default=0.001, help='minimum lr for weights')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum for weights')
        parser.add_argument('--weight_decay', type=float, default=3e-4,
                            help='weight decay for weights')
        parser.add_argument('--grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        # ================= dataset settings ==================
        parser.add_argument('--dataset', type=str, default='cifar10', help='CIFAR10')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--train_portion', type=float, default=0.9, help='portion of training data')
        parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
        parser.add_argument('--advanced', action='store_true', help='advanced data transform. apply resize (224,224)')
        # ================= training settings ==================
        parser.add_argument('--epochs', type=int, default=50, help='# of training epochs')
        parser.add_argument('--T', type=float, default=10, help='temperature of softmax with temperature')
        parser.add_argument('--l', type=float, default=0.5, help='ratio between soft target loss and hard target loss')
        parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
        parser.add_argument('--nonkd', action='store_true', help='execute KD learning')
        # ================= description ==================
        parser.add_argument('--description', type=str, default='', help='experiment details')
        # ================= others ==================        
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--seed', type=int, default=0, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')

        return parser
    
    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = '../data/'

        self.genotype = gt.from_str(self.genotype)
        self.DAG = gt.from_str(self.DAG)
        
        self.path = os.path.join(f'results/evaluate_stage_KD/{self.dataset}', self.name)
        self.exp_name = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
        self.path = os.path.join(self.path, self.exp_name)

        self.gpus = parse_gpus(self.gpus)