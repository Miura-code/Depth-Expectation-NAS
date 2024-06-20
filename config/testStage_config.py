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


class TestStageConfig(BaseConfig):
    def build_parser(self):
        # ===========================================cifar10==========================================
        parser = get_parser("Test final model of H^s-DAS config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', type=str, default='cifar10', help='CIFAR10')
        parser.add_argument('--batch_size', type=int, default=96, help='batch size')
        parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--init_channels', type=int, default=36)
        parser.add_argument('--layers', type=int, default=20, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--aux_weight', type=float, default=0.0, help='auxiliary loss weight')
        parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')

        parser.add_argument('--genotype', required=True, help='Cell genotype')
        parser.add_argument('--DAG', required=True, help='DAG genotype')
        parser.add_argument('--dist', action='store_true', help='use multiprocess_distributed training')
        parser.add_argument('--local_rank', default=0)
        parser.add_argument('--resume_path', type=str, default=None)
        parser.add_argument('--exclude_bias_and_bn', type=bool, default=True)

        parser.add_argument('--train_portion', type=float, default=1.0, help='portion of training data')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()

        super().__init__(**vars(args))

        self.data_path = '../data/'
        
        directory, _ = os.path.split(args.resume_path)
        directory = directory.rstrip(os.path.sep)
        self.path = os.path.join(directory, "test")

        self.genotype = gt.from_str(self.genotype)
        self.DAG = gt.from_str(self.DAG)
        self.gpus = parse_gpus(self.gpus)
        self.amp_sync_bn = True
        self.amp_opt_level = "O0"

        self.path = '{}/{}-{}'.format(self.path, args.name, time.strftime("%Y%m%d-%H%M%S"))
        # utils.create_exp_dir(args.path, scripts_to_save=None)
