import tensorflow as tf
import numpy as np

class TFEffect:
    def __init__(self, effect_idx, parameters, total_effects, effect_name, total_params, param_idxs, order):
        # Create a representation of the effect
        if effect_idx == -1:
            self.one_hot = tf.zeros(total_effects, dtype=tf.float32)
            self.param_repr = tf.zeros(total_params, dtype=tf.float32)
            self.effect_name = "None"
            return
        
        self.total_effects = total_effects
        # Create a one hot encoding of the effect
        one_hot = [0] * self.total_effects
        one_hot[effect_idx] = 1
        self.one_hot = tf.convert_to_tensor(one_hot, dtype=tf.float32)
        
        # Create a representation of the parameters
        self.total_params = total_params
        self.param_repr = tf.zeros(total_params, dtype=tf.float32)
        # the location of each effect's parameters will be predefined with param_idxs
        self.param_repr = tf.tensor_scatter_nd_update(self.param_repr, tf.expand_dims(param_idxs, 1), tf.convert_to_tensor(parameters, dtype=tf.float32))
        self.param_repr = tf.convert_to_tensor(self.param_repr, dtype=tf.float32)
        
        # Store the effect name and order
        self.effect_name = effect_name
        self.order = order
        
    def to_dict(self):
        return {
            "effect": self.effect_name,
            "one_hot": self.one_hot,
            "parameters": self.param_repr,
            "order": self.order
        }

class TFEffectsChain:
    def __init__(self, effects, dry_tone_path, wet_tone_path, total_effects, max_chain_length, total_params):
        assert len(effects) <= total_effects, "The number of effects in the chain must be less than or equal to the maximum number of effects"
        
        # Pad effects chain to be the length of max_chain_length
        if len(effects) < max_chain_length:
            for i in range(max_chain_length - len(effects)):
                effects.append(TFEffect(-1, None, total_effects, "None", total_params, None, len(effects) + i))
        
        # Stack effects and parameters into tensors
        self.effects = tf.stack([effect.one_hot for effect in effects])
        self.parameters = tf.stack([effect.param_repr for effect in effects])
        
        # Keep track of the names of each effect
        self.names = [effect.effect_name for effect in effects]
        self.dry_tone_path = dry_tone_path
        self.wet_tone_path = wet_tone_path
        
    def __len__(self):
        return len(self.effects)
    
    def to_dict(self):
        return {
            "effects": self.effects,
            "parameters": self.parameters,
            "dry_tone_path": self.dry_tone_path,
            "wet_tone_path": self.wet_tone_path
        }

class TFEffectChainDataset(tf.data.Dataset):
    def __init__(self, data):
        """
        Pass in a list of TFEffectsChain objects
        Note: This is a basic implementation. For production use, you might want to
        use tf.data.Dataset.from_tensor_slices() or other more optimized methods.
        """
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def get_item(entry):
        """
        Return the wet tone data for a given entry
        """
        return {
            "dry_tone_path": entry['dry_tone_path'],
            "wet_tone_path": entry['wet_tone_path'],
            "wet_tone_features": entry['wet_tone_features'],
            "effect_names": entry['names'],
            "effects": entry['effects'],
            "parameters": entry['parameters'],
            "index": entry['index']
        }
    
    def as_dataset(self):
        """
        Convert to a proper tf.data.Dataset
        """
        # Convert the data to the format expected by TensorFlow
        dataset_dicts = [item.to_dict() for item in self.data]
        
        # Create a tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices(dataset_dicts)
        return dataset.map(self.get_item) 