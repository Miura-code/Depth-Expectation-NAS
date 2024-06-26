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


class TestTeacherConfig(BaseConfig):
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
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--cutout_length', type=int, default=0, help='cutout length')
        parser.add_argument('--model_name', type=str, default='densenet121', help='teacher model name')

        parser.add_argument('--resume_path', type=str, default=None)

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

        self.gpus = parse_gpus(self.gpus)
        self.amp_sync_bn = True
        self.amp_opt_level = "O0"

        self.path = '{}/{}-{}'.format(self.path, args.name, time.strftime("%Y%m%d-%H%M%S"))
