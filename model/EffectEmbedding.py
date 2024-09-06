import torch
import torch.nn as nn
import math

class CrossAttentionFeatureFusion(nn.Module):
    def __init__(self, n_heads, embedding_dims=768):
        '''
        Attention layer to combine the features of effects and parameters
        Learns a query value on the effect and a key value on the parameters to learn important features
        Uses cross attention to combine the features
        '''
        super(CrossAttentionFeatureFusion, self).__init__()
        self.embedding_dims = embedding_dims

        self.query = nn.Linear(embedding_dims, embedding_dims)
        self.key = nn.Linear(embedding_dims, embedding_dims)
        self.value = nn.Linear(embedding_dims, embedding_dims)
        self.softmax = nn.Softmax(dim=1)
        return
    
    def forward(self, effect, parameters):
        query = self.query(effect)
        key = self.key(parameters)
        value = self.value(parameters)
        attention = self.softmax(torch.matmul(query, key.T) / math.sqrt(self.embedding_dims))
        attn_output = torch.matmul(attention, value)
        return attn_output

class EffectEmbedding(nn.Module):
    def __init__(self, max_effects, max_effect_params, num_params,attn_heads=8, embedding_dim=768):
        '''
        Create an Embedding Representation of an Effect and its Parameters
        Inputs: 
        - effect: one hot encoded representation of effect
        - effect_params: a vector of parameters for the effect of length max_params
        - num_params: the number of parameters used the effect, used to make the effect mask
        '''
        super(EffectEmbedding, self).__init__()
        
        self.num_params = num_params
        self.max_params = max_effect_params
        self.max_effects = max_effects
        self.attention = CrossAttentionFeatureFusion(embedding_dim)
        self.multihead_attention = nn.MultiheadAttention(embedding_dim, num_heads=attn_heads)
        # Create embedding from one-hot encoded Effect
        self.effect_embed = nn.Linear(self.max_effects, embedding_dim)
        # Create a mask for the effect parameters
        self.param_mask = torch.tensor([1 if i < num_params else 0 for i in range(self.max_params)])
        # Create a positional encoding for the effect
        self.positional_encoding = nn.Linear(self.max_effects, embedding_dim)
        # Create embedding from effect parameters
        self.param_embed = nn.Linear(self.max_params, embedding_dim)
        # Create a 
        return



    def forward(self, effect, parameters):
        # Embed the effect
        effect = self.effect_embed(effect)
        # Embed the parameters
        parameters = self.param_embed(parameters)
        # Mask the parameters
        parameters = parameters * self.param_mask
        # Add the positional encoding
        effect = effect + self.positional_encoding(effect)
        # Fuse the effect and parameters with cross attention
        attn_embeds = self.cross_attention(effect, parameters)
        return attn_embeds