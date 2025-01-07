from copy import deepcopy
import torch
import torch.nn as nn

class Feature_Extractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        
        self.model = deepcopy(model)

    def forward(self, x):
        s0 = s1 = self.model.stem(x)
        aux_logits = None
        s0 = s1 = self.model.bigDAG1(s0, s1)
        stage1_features = s0 = s1 = self.model.cells[6](s0, s1) # reduction
        s0 = s1 = self.model.bigDAG2(s0, s1) 
        stage2_features = s0 = s1 = self.model.cells[13](s0, s1) # reduction
        aux_logits = self.model.aux_head(s1)
        stage3_features = s0 = s1 = self.model.bigDAG3(s0, s1)

        out = self.model.gap(s1)
        out = out.view(out.size(0), -1)
        logits = self.model.linear(out)

        extracted_features = {
            "stage1": stage1_features,
            "stage2": stage2_features,
            "stage3": stage3_features,
            "logits": logits
        }
        return extracted_features
    
class Feature_Extractor_searched(nn.Module):
    def __init__(self, model):
        super().__init__()
        
        self.model = deepcopy(model)
    
    def forward(self, x, weights_DAG):
        s0 = s1 = self.model.stem(x)
        s0 = s1 = self.model.bigDAG1(s0, s1, weights_DAG[0 * self.model.n_big_nodes: 1 * self.model.n_big_nodes])
        stage1_features = s0 = s1 = self.model.cells[1 * self.model.n_layers // 3](s0, s1)
        
        s0 = s1 = self.model.bigDAG2(s0, s1, weights_DAG[1 * self.model.n_big_nodes: 2 * self.model.n_big_nodes])
        stage2_features = s0 = s1 = self.model.cells[2 * self.model.n_layers // 3](s0, s1)
        
        stage3_features = s0 = s1 = self.model.bigDAG3(s0, s1, weights_DAG[2 * self.model.n_big_nodes: 3 * self.model.n_big_nodes])
        
        out = self.model.gap(s1)
        out = out.view(out.size(0), -1)
        logits = self.model.linear(out)
        
        extracted_features = {
            "stage1": stage1_features,
            "stage2": stage2_features,
            "stage3": stage3_features,
            "logits": logits
        }
        return extracted_features