
import json
import os
from icecream import ic

import torch
from models.augment_stage import AugmentStage
import utils
from utils.data_util import get_data, split_dataloader
from utils.graphs import find_unreachable_nodes, make_StageGraph
from utils.measurement_utils import count_ModelSize_exclude_extra
from utils.parser import get_parser, parse_gpus, BaseConfig
import genotypes.genotypes as gt

class Config(BaseConfig):
    def build_parser(self):
        parser = get_parser("Calculate network macs config")
        # ======== model settings ============
        parser.add_argument('--genotype', required=True, help='Cell genotype')
        parser.add_argument('--DAG', required=True, help='DAG genotype')
        parser.add_argument('--init_channels', type=int, default=32)
        parser.add_argument('--layers', type=int, default=20, help='# of layers')
        parser.add_argument('--spec_cell', action='store_true', help='Use stage specified cell architecture at each stage')
        parser.add_argument('--aux_weight', type=float, default=0.4, help='auxiliary loss weight')
        parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path prob')
        # ======== dataset settings ============
        parser.add_argument('--dataset', type=str, default='cifar10', help='CIFAR10')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
        parser.add_argument('--advanced', action='store_true', help='advanced data transform. apply resize (224,224)')
        parser.add_argument('--train_portion', type=float, default=1.0, help='portion of training data')
        # ======== other settings ============
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--seed', type=int, default=0, help='random seed')

        return parser
    
    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = '../data/'
        self.genotype = gt.from_str(self.genotype)
        self.DAG = gt.from_str(self.DAG)

        self.gpus = parse_gpus(self.gpus)
        self.device = utils.set_seed_gpu(self.seed, self.gpus)


def main(config):

    input_size, input_channels, n_classes, train_data, valid_data = get_data(
        config.dataset, config.data_path, config.cutout_length, validation=True, advanced=config.advanced
    )
    train_loader, valid_loader = split_dataloader(train_data, config.train_portion, config.batch_size, config.workers)
    use_aux = config.aux_weight > 0.

    model = AugmentStage(input_size, input_channels, config.init_channels, n_classes, config.layers, use_aux, config.genotype, config.DAG, spec_cell=config.spec_cell)
    model = model.to(config.device)

    last_node = len(config.DAG.DAG1)+2

    G = make_StageGraph(config.DAG.DAG1, config.DAG.DAG1_concat)
    dead_cell1 = ic(find_unreachable_nodes(G=G, last_node=last_node))
    G = make_StageGraph(config.DAG.DAG2, config.DAG.DAG2_concat)
    dead_cell2 = ic(find_unreachable_nodes(G=G, last_node=last_node))
    G = make_StageGraph(config.DAG.DAG3, config.DAG.DAG3_concat)
    dead_cell3 = ic(find_unreachable_nodes(G=G, last_node=last_node))

    # mac, params = utils.measurement_utils.count_ModelSize_byptflops(model, (3,32,32) if config.advanced else (3,16,16))
    # print("param size = {}MB, mac = {}".format(params, mac))
    
    inputs = torch.randn(1,3,32,32) if config.advanced else torch.randn(1,3,16,16)
    inputs = inputs.to(config.device)
    mac, params, ret_dict = utils.measurement_utils.count_ModelSize_bythop(model, inputs)
    print("param size = {}MB, mac = {}".format(params, mac))

    ignore_layer = [
        "stem0", "stem1", "cells"
    ]
    mac, params = count_ModelSize_exclude_extra(mac, params, ret_dict, ignore_layer, [dead_cell1, dead_cell2, dead_cell3])
    print("param size = {}MB, mac = {}".format(params, mac))


if __name__ == "__main__":
    config = Config()
    main(config)
