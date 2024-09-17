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


class GraphConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Search Dag config")
        # ================= file settings ==================
        parser.add_argument('--name')
        parser.add_argument('--save', type=str, default='EXP', help='experiment name')
        # ================= model settings ==================
        parser.add_argument('--genotype', help='Cell genotype')
        parser.add_argument('--DAG_name', help='Stage genotype')
        # ================= details ==================
        parser.add_argument('--description', type=str, default='', help='experiment details')
        # ================= others ==================
        parser.add_argument('--seed', type=int, default=0, help='ramdom seed')
        return parser
    
    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        if self.genotype is not None:
            self.genotype = gt.from_str(self.genotype)
        if self.DAG_name is not None:
            self.DAG = gt.from_str(self.DAG_name)
