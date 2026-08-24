import torch.nn as nn
import torch.nn.functional as F
from . import model_config

class MLP(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        hidden = int(8/3) * model_config.d_model
        self.gate = nn.Linear(model_config.d_model, hidden, bias = False)
        self.up_proj = nn.Linear(model_config.d_model, hidden, bias = False)
        self.down_proj = nn.Linear(hidden, model_config.d_model, bias = False)

    def forward(self,x):
        return self.down_proj(self.up_proj(x) * F.silu(self.gate(x)))

class SummarizationModel(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self,x):
        ...