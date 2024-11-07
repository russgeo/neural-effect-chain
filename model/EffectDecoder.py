from transformers import ASTModel, AutoFeatureExtractor, AutoConfig
import librosa
import numpy as np
import torch.nn as nn
from torch import softmax
import os

class EffectDecoder(nn.Module):
    
    def __init__(self, num_effects, attn_heads=8, embedding_dim=768, **kwargs):
        '''
        Create an Embedding Representation of an Effect and its Parameters
        Inputs: 
        - effect: one hot encoded representation of effect
        - effect_params: a vector of parameters for the effect of length max_params
        - num_params: the number of parameters used the effect, this will be the length of the output vector
        '''
        super(EffectDecoder, self).__init__()
        self.spectrogram_embedding_model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.attn_heads = attn_heads
        # fine tune ontop of pretrained model
        self.input_embedding = nn.Linear(embedding_dim,embedding_dim)
        self.target_embedding = nn.Linear(embedding_dim,embedding_dim)
        # attn layer
        self.cross_attn = nn.MultiheadAttention(embedding_dim,attn_heads)
        #feed forward layer
        self.feed_forward = nn.Linear(embedding_dim, embedding_dim)
        self.cls_effect = nn.Linear(embedding_dim, num_effects)
        self.softmax = softmax
        return
    
    def forward(self, input_features,input_f0,input_loudness, target_features, target_f0, target_loudness):
        '''
        Forward pass of the model
        Inputs:
        - input_spectrogram: the input spectrogram
        - input_f0: the input f0
        - target_spectrogram: the target spectrogram
        - target_f0: the target f0
        Outputs:
        - output: the output of the model
        '''
        input_embedding = self.input_embedding(self.spectrogram_embedding_model(**input_features).pooler_output)
        target_embedding = self.target_embedding(self.spectrogram_embedding_model(**target_features).pooler_output)
        
        attn_output = self.cross_attn(input_embedding, target_embedding, target_embedding)
        # return the output of the feed forward layer to predict parameters
        output = self.feed_forward(attn_output[0])
        # predict the effect
        effect = self.softmax(self.cls_effect(output),dim=-1)
        return output, effect
