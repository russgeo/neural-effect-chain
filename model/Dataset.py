import torch
import numpy as np
from torch.utils.data import Dataset

class Effect():
    def __init__(self, effect_idx, parameters, max_effects, effect_name, total_params, param_idxs):
        # Create a representation of the effect
        # maximum amount of effects that can be used in the effects chain - also the length of the one hot encoding
        if effect_idx == -1:
            self.one_hot = torch.zeros(max_effects, dtype=torch.float32)
            self.param_repr = torch.zeros(total_params, dtype=torch.float32)
            self.effect_name = "None"
            return
        self.max_effects = max_effects
        # Create a one hot encoding of the effect
        one_hot = [0] * self.max_effects
        one_hot[effect_idx] = 1
        self.one_hot = torch.tensor(one_hot, dtype=torch.float32)
        # Create a representation of the parameters
        # Total number of parameters = sum of the number of parameters for each effect
        self.total_params = total_params
        self.param_repr = [0] * total_params
        # the location of each effect's parameters will be predefined with param_idxs
        self.param_repr[param_idxs[0]:param_idxs[-1]+1] = parameters
        self.param_repr = torch.tensor(self.param_repr, dtype=torch.float32)
        # Store the effect name
        self.effect_name = effect_name
        return
    
    def __dict__(self):
        return {"Effect": self.effect_name,"One hot":self.one_hot, "Parameters": self.param_repr}
        
    
    
class EffectsChain():
    def __init__(self, effects, dry_tone_path, wet_tone_path, max_effects,total_params):
        assert len(effects) <= max_effects, "The number of effects in the chain must be less than or equal to the maximum number of effects"
        # # pad effects chaain to be the length of max_effects
        if len(effects) < max_effects:
            for i in range(max_effects-len(effects)):
                effects.append(Effect(-1, None, max_effects, "None", total_params, None))
        
        self.effects = torch.stack([effect.one_hot for effect in effects])
        self.parameters = torch.stack([effect.param_repr for effect in effects])
        self.names = [effect.effect_name for effect in effects]
        self.dry_tone_path = dry_tone_path
        self.wet_tone_path = wet_tone_path
        return
    
    def __len__(self):
        return len(self.effects)

class EffectChainDataset(Dataset):
    def __init__(self, data):
        '''
        
        '''
        self.data = data
        return
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, wet_tone_path):
        '''
        Return the effects chain for a given wet tone path
        '''
        return self.data[wet_tone_path]