
__all__ = [
    "h_das_224baseline",
    "h_das_224custom"
    # "H-DAS-Cell",
    # "DARTS"
]

from copy import deepcopy
from typing import Any
import torch
import torch.nn as nn

from models.augment_stage import AugmentStage
import genotypes.genotypes as gt

class H_DAS_STAGE(AugmentStage):
    def __init__(self, input_size, C_in, C, n_classes, n_layers, auxiliary, genotype,
                 DAG, stem_multiplier=4, cell_multiplier=4, spec_cell=False, **kwargs):
        super().__init__(input_size, C_in, C, n_classes, n_layers, auxiliary, genotype,
                 DAG, stem_multiplier, cell_multiplier, spec_cell, **kwargs)
       
    def get_classifier(self):
        return self.linear
    
    def get_head(self):
        return self.get_classifier()
    
    def get_features(self):
        return [self.bigDAG1, self.bigDAG2, self.bigDAG3]

def _h_das_stage(
    input_size, 
    C_in, 
    C, 
    n_classes, 
    n_layers, 
    auxiliary, 
    genotype,
    DAG, 
    stem_multiplier=4, 
    cell_multiplier=4, 
    spec_cell=False,
    **kwargs: Any,
) -> H_DAS_STAGE:
    
    model = H_DAS_STAGE(input_size, 
    C_in, 
    C, 
    n_classes, 
    n_layers, 
    auxiliary, 
    genotype,
    DAG, 
    stem_multiplier, 
    cell_multiplier, 
    spec_cell,
    **kwargs)

    return model

def h_das_224custom(num_classes, getnoytpe, DAG, **kwargs: Any) -> H_DAS_STAGE:
    r"""H-DAS from
    `Unchain the Search Space with Hierarchical Differentiable Architecture Search <https://ojs.aaai.org/index.php/AAAI/article/view/17048>`_.

    Args:
        **kwargs: parameters passed to the H_DAS_STAGE
    """

    return _h_das_stage(224, 3, 32, num_classes, 20, True, getnoytpe, DAG, spec_cell=True, **kwargs)

def h_das_224baseline(num_classes, **kwargs: Any) -> H_DAS_STAGE:
    r"""H-DAS from
    `Unchain the Search Space with Hierarchical Differentiable Architecture Search <https://ojs.aaai.org/index.php/AAAI/article/view/17048>`_.
    input size=224,
    stage learned without depthloss
    cell is the best archteture
    Args:
        **kwargs: parameters passed to the H_DAS_STAGE
    """
    
    genotype = gt.from_str("BASELINE_BEST")
    DAG = gt.from_str("NoDepthLoss_BEST")
        
    return _h_das_stage(224, 3, 32, num_classes, 20, True, genotype, DAG, spec_cell=True, **kwargs)


