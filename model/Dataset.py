import torch
import numpy as np
from torch.utils.data import Dataset

class Effect():
    def __init__(self, effect_idx, parameters, max_effects, effect_name, total_params, param_idxs):
        # Create a representation of the effect
        # maximum amount of effects that can be used in the effects chain - also the length of the one hot encoding
        self.max_effects = max_effects
        # Create a one hot encoding of the effect
        one_hot = [0]* self.max_effects
        one_hot[effect_idx] = 1
        self.one_hot = torch.tensor(one_hot, dtype=torch.float32)
        # Create a representation of the parameters
        # Total number of parameters = sum of the number of parameters for each effect
        self.total_params = total_params
        self.param_repr = [0] * (total_params-1)
        # the location of each effect's parameters will be predefined with param_idxs
        self.param_repr[param_idxs[0]:param_idxs[-1]] = parameters
        self.param_repr = torch.tensor(self.param_repr, dtype=torch.float32)
        # Store the effect name
        self.effect_name = effect_name
        return
    
    def __dict__(self):
        return {"Effect": self.effect_name,"One hot":self.one_hot, "Parameters": self.param_repr}
        
    
    
class EffectsChain(Dataset):
    def __init__(self, effects, dry_tone_path, wet_tone_path):
        self.effects = effects
        self.dry_tone_path = dry_tone_path
        self.wet_tone_path = wet_tone_path
        return
    
    def __len__(self):
        return len(self.effects)
    
    def __getitem__(self, idx):
        return {"effect":self.effects[idx].one_hot,"order": idx,"parameters":self.effects[idx].param_repr,"dry_tone_path":self.dry_tone_path,"wet_tone_path":self.wet_tone_path}
    
if __name__ == "__main__":
    # example of how to use the Effect and EffectsChain classes
    # This dictionary will be predefined when the dataset is generated
    effect_to_idx = {"Reverb": 0, "Delay": 1, "Distortion": 2}
    # each instance of an effect will look like this
    reverb = Effect(effect_to_idx['Reverb'], [1,2,3], 3, "Reverb", 10, [0,1,2,3])
    delay = Effect(effect_to_idx['Delay'], [4,5,6], 3, "Delay", 10, [4,5,6])
    distortion = Effect(effect_to_idx['Distortion'], [7,8,9], 3, "Distortion", 10, [7,8,9])
    # 
    effects = [reverb, delay, distortion]
    dataset = EffectsChain(effects, "dry.wav", "wet.wav")
    print(dataset[2])