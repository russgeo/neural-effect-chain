import torch
import torch.nn as nn
import math
from Dataset import Effect, EffectsChain
class CrossAttentionFeatureFusion(nn.Module):
    def __init__(self, n_heads, embedding_dims=768):
        '''
        Attention layer to combine the features of effects and parameters
        Learns a query value on the effect and a key value on the parameters to learn important features
        Uses cross attention to combine the features
        '''
        super(CrossAttentionFeatureFusion, self).__init__()
        self.embedding_dims = embedding_dims
        assert embedding_dims % n_heads == 0
        self.head_dim = embedding_dims // n_heads

        self.query = nn.Linear(embedding_dims, embedding_dims)
        self.key = nn.Linear(embedding_dims, embedding_dims)
        self.value = nn.Linear(embedding_dims, embedding_dims)
        self.softmax = nn.Softmax(dim=-1)

        self.n_heads = n_heads
        return
    
    def forward(self, effect, parameters):
        batch_size, seq_len = effect.shape
        query = self.query(effect)
        key = self.key(parameters)
        value = self.value(parameters)
        # Split the heads for multihead attention
        query = query.reshape(batch_size, self.head_dim,self.n_heads)
        key = key.reshape(batch_size, seq_len,self.n_heads,self.head_dim)
        value = value.reshape(batch_size, seq_len / self.n_heads,self.n_heads,self.head_dim)
        # Compute the attention
        attention = self.softmax(torch.matmul(query, key.reshape(-1)) / math.sqrt(self.embedding_dims))
        attn_output = torch.matmul(attention, value)
        return attn_output
    
def positionalencoding1d(d_model, length):
    """
    Source: https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py 
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe 

    

class EffectEmbedding(nn.Module):
    def __init__(self, max_effects, max_effect_params, num_params,seq_order,attn_heads=8, embedding_dim=768):
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
        self.pos_encodings = positionalencoding1d(embedding_dim, max_effects)
        self.seq_order = seq_order
        self.embedding_dim = embedding_dim
        self.cross_attention = CrossAttentionFeatureFusion(attn_heads,embedding_dim)
        # Create embedding from one-hot encoded Effect
        self.effect_embed = nn.Linear(self.max_effects, embedding_dim)
        # Create embedding from effect parameters
        self.param_embed = nn.Linear(self.max_params, embedding_dim)
        # Create a final linear layer to output the embedding
        self.final = nn.Linear(embedding_dim, embedding_dim)
        return
    
    def forward(self, effect, parameters):
        # Embed the effect
        effect = self.effect_embed(effect)
        # Embed the parameters
        parameters = self.param_embed(parameters)
        # Mask the parameters
        #parameters = parameters * self.param_mask
        # Add the positional encoding
        effect = effect + self.pos_encodings[self.seq_order]
        # Fuse the effect and parameters with cross attention
        attn_embeds = self.cross_attention(effect, parameters)
        out = self.final(attn_embeds)
        return out
    
if __name__ == "__main__":
    # Test the Effect Embedding
    max_effects = 3
    max_params = 10
    reverb = Effect(0, [1,2,3], max_effects, "Reverb", max_params, [0,1,2])
    delay = Effect(1, [4,5,6], max_effects, "Delay", max_params, [3,4,5])
    chain = EffectsChain([reverb, delay], "dry.wav", "wet.wav")
    effect_embedding = EffectEmbedding(max_effects, max_params, 3, 1)
    out = effect_embedding(chain[0]['effect'], chain[0]['parameters'])
    print(out.shape)
    print(out)