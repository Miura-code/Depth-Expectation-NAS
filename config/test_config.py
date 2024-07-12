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


class TestConfig(BaseConfig):
    def build_parser(self):
        # ===========================================cifar10==========================================
        parser = get_parser("Test final model of H^s-DAS config")
        # ================= file settings ==================
        parser.add_argument('--save', required=True)
        # ================= dataset settings ==================
        parser.add_argument('--dataset', type=str, default='cifar10', help='CIFAR10')
        parser.add_argument('--batch_size', type=int, default=96, help='batch size')
        parser.add_argument('--cutout_length', type=int, default=0, help='cutout length')
        parser.add_argument('--train_portion', type=float, default=1.0, help='portion of training data')
        parser.add_argument('--advanced', action='store_true', help='advanced data transform. apply resize (224,224)')
        # ================= model settings ==================
        parser.add_argument('--model_name', type=str, default=None, help='teacher model name for testing finetuned teacher')
        parser.add_argument('--genotype', type=str, default=None, help='Cell genotype for testing searched cell architecture')
        parser.add_argument('--resume_path', type=str, default=None)
        parser.add_argument('--init_channels', type=int, default=32)
        parser.add_argument('--layers', type=int, default=20, help='# of layers')
        parser.add_argument('--aux_weight', type=float, default=0.4, help='auxiliary loss weight for testing searched cell architecture')
        # ================= test settings ==================
        parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
        # ================= description ==================
        parser.add_argument('--description', type=str, default='', help='experiment details')
        # ================= others ==================
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')        
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()

        super().__init__(**vars(args))

        self.data_path = '../data/'
        
        directory, _ = os.path.split(args.resume_path)
        directory = directory.rstrip(os.path.sep)
        self.path = os.path.join(directory, "test")

        if self.genotype is not None:
            self.genotype = gt.from_str(self.genotype)

        self.gpus = parse_gpus(self.gpus)
        self.amp_sync_bn = True
        self.amp_opt_level = "O0"

        self.path = '{}/{}-{}'.format(self.path, args.save, time.strftime("%Y%m%d-%H%M%S"))

        self.cifar = True if "cifar" in args.dataset else False
