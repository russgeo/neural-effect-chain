from transformers import ASTModel, AutoFeatureExtractor, AutoConfig
import librosa
import numpy as np
import torch.nn as nn
from torch import softmax
import os
import torch

class ParameterDecoder(nn.Module):
    '''
    Basic parameter decoder
    '''
    def __init__(self, total_params,effect_to_param_mask, effect,embedding_dim=768):
        super(ParameterDecoder, self).__init__()
        self.total_params = total_params
        self.effect = effect
        self.param_mask = torch.tensor(effect_to_param_mask[effect])
        self.embedding = nn.Linear(embedding_dim,embedding_dim)
        self.output = nn.Linear(embedding_dim, total_params)
        return
    
    def forward(self, input_features):
        embedding = self.embedding(input_features)
        output = self.output(embedding)
        masked_output = output * self.param_mask
        return masked_output