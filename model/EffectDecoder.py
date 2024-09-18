from transformers import ASTModel, AutoFeatureExtractor, AutoConfig
import librosa
import numpy as np
import torch.nn as nn
import os

class EffectDecoder(nn.Module):
    
    def __init__(self,input_sr,target_sr, num_effects, attn_heads=8, embedding_dim=768, **kwargs):
        '''
        Create an Embedding Representation of an Effect and its Parameters
        Inputs: 
        - effect: one hot encoded representation of effect
        - effect_params: a vector of parameters for the effect of length max_params
        - num_params: the number of parameters used the effect, used to make the effect mask
        '''
        super(EffectDecoder, self).__init__()
        self.spectrogram_embedding_model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.embedding_dim = embedding_dim
        self.attn_heads = attn_heads
        self.input_sr = input_sr
        self.target_sr = target_sr
        self.embedding = nn.Linear(embedding_dim,embedding_dim)
        self.cross_attn = nn.MultiheadAttention(embedding_dim,attn_heads)
        self.cls_head = nn.Linear(embedding_dim, num_effects)
        return
    
    def forward(self, input_features, target_features):
        '''
        Forward pass of the model
        Inputs:
        - input_spectrogram: the input spectrogram
        - target_spectrogram: the target spectrogram
        Outputs:
        - output: the output of the model
        '''
        input_embedding = self.spectrogram_embedding_model(**input_features).pooler_output
        target_embedding = self.spectrogram_embedding_model(**target_features).pooler_output
        attn_output = self.cross_attn(input_embedding, target_embedding, target_embedding)
        output = self.cls_head(attn_output[0])
        return output
