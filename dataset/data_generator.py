from pedalboard import Pedalboard, Delay, Gain, Chorus, Reverb, Distortion, Compressor, Mix, Phaser, NoiseGate, PitchShift, PeakFilter, LowpassFilter, LowShelfFilter, Limiter, LadderFilter, IIRFilter, HighpassFilter, HighShelfFilter, GSMFullRateCompressor, Convolution, Clipping, Invert
from pedalboard.io import ReadableAudioFile
import random
import pickle
import tqdm
from transformers import AutoFeatureExtractor
import os
from dataset.dataset import Effect, EffectsChain, EffectChainDataset
#from dataset.tf_dataset import TFEffectChainDataset, TFEffectsChain, TFEffect
#from model.feature_extractor import FeatureExtractor
import numpy as np
from scipy.stats import truncnorm
# Create a list of all the effects
effects = [Delay, Gain, Chorus, Reverb, Distortion, Compressor, Phaser, NoiseGate, PitchShift, PeakFilter, LowpassFilter, LowShelfFilter, Limiter, LadderFilter, HighpassFilter, HighShelfFilter, Clipping]
#import tensorflow as tf
# Create -a mapping of effect names to their parameters and max/min values of said parameters
#TODO: adjust max min values to more accurate values
effects_parameters = {
    "Reverb": {
        "room_size": (0, 1),
        "damping": (0, 1),
        "wet_level": (0, 1),
        "dry_level": (0, 1),
        "width": (0, 1),
        "freeze_mode": (0, 1)
    },
    "Delay": {
        "delay_seconds": (0, 2),
        "feedback": (0, 1),
        "mix": (0, 1)
    },
    "Gain": {
        "gain_db": (-60, 24)
    },
    "Chorus": {
        "rate_hz": (0.1, 5.0),
        "depth": (0, 1),
        "centre_delay_ms": (0, 50),
        "feedback": (-1, 1),
        "mix": (0, 1)
    },
    "Distortion": {
        "drive_db": (0, 60)
    },
    "Compressor": {
        "threshold_db": (-100, 0),
        "ratio": (1, 20),
        "attack_ms": (0.1, 100),
        "release_ms": (10, 1000)
    },
    "Phaser": {
        "rate_hz": (0.01, 10),
        "depth": (0, 1),
        "centre_frequency_hz": (0, 2000),
        "feedback": (-1, 1),
        "mix": (0, 1)
    },
    "NoiseGate": {
        "threshold_db": (-100, 0),
        "attack_ms": (0.1, 100),
        "release_ms": (10, 1000)
    },
    "PitchShift": {
        "semitones": (-12, 12),
    },
    "PeakFilter": {
        "cutoff_frequency_hz": (20, 20000),
        "gain_db": (-24, 24),
        "q": (0.1, 10)
    },
    "LowpassFilter": {
        "cutoff_frequency_hz": (20, 20000)
    },
    "LowShelfFilter": {
        "cutoff_frequency_hz": (20, 20000),
        "gain_db": (-24, 24),
        "q": (0.1, 10)
    },
    "Limiter": {
        "threshold_db": (-100, 0),
        "release_ms": (10, 1000)
    },
    "LadderFilter": {
        "cutoff_hz": (20, 20000),
        "resonance": (0.0, 1)
    },
    "HighpassFilter": {
        "cutoff_frequency_hz": (20, 20000),
    },
    "HighShelfFilter": {
        "cutoff_frequency_hz": (20, 20000),
        "gain_db": (-24, 24),
        "q": (0.1, 10)
    },
    "Clipping": {
        "threshold_db": (-1, 1)
    }
    }

class DataGenerator():
    def __init__(self, 
                 effects_to_parameters: dict,
                 effects: list) -> None:
        self.effects_to_parameters = effects_to_parameters
        # calculate the total number of possible parameters
        total_parameters = 0
        for effect, params in effects_parameters.items():
            total_parameters += len(params)
        self.total_parameters = total_parameters
        # Create a dictionary to store the indices of the parameters for each effect
        effect_to_param_indices = {}
        current_index = 0
        for effect, params in effects_parameters.items():
            num_params = len(params)
            if num_params > 0:
                effect_to_param_indices[effect] = list(range(current_index, current_index + num_params))
                current_index += num_params
        self.effect_to_param_indices = effect_to_param_indices
        # map each effect to a one hot encoding index
        self.effect_to_index = {effect.__name__: i for i, effect in enumerate(effects)}
        self.effects = effects
        return
    def create_data(self,
                    num_samples: int, 
                    dry_tone_dir: str,
                    dry_tones: list, 
                    max_chain_length: int,
                    feature_extractor=AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593"),
                    sample_rate=16000,
                    dry_tone_features=True
                    ) -> EffectChainDataset:
        # Create an empty list to store the data
        data = []
        for dry_tone_path in tqdm.tqdm(dry_tones):
            name,ext = dry_tone_path.split('.')
            dry_tone_path = os.path.join(dry_tone_dir, dry_tone_path)
            with ReadableAudioFile(dry_tone_path) as f:
                # re sample the audio file to match the sample rate, pretrained model is sampled at 16000
                re_sampled = f.resampled_to(sample_rate)
                dry_tone = re_sampled.read(int(sample_rate * f.duration))
                re_sampled.close()
                f.close()
            # Loop over the number of samples
            if dry_tone_features:
                dry_tone_features = feature_extractor(dry_tone,sample_rate,return_tensors='pt')
            for i in range(num_samples):
                wet_tone_data = {}
                wet_tone_data['dry_tone_path'] = dry_tone_path
                wet_tone_data['wet_tone_path'] = f'data/wet_tones/{name}_wet_{i}.{ext}'
                # Create a new pedalboard
                pedalboard = Pedalboard()
                # Randomly select a number of effects to add to the pedalboard
                num_effects = random.randint(1, max_chain_length)
                # Create a dictionary to the effects used and their parameters
                effect_list = []
                for j in range(num_effects):
                    # Randomly select an effect to add
                    effect = random.choice(self.effects)
                    # Get the effect name
                    effect_name = effect.__name__
                    # Get the effect parameters
                    parameters = self.effects_to_parameters[effect_name].keys()
                    # Create a dictionary to store the effect data (parameters and order)
                    effect_data = {}
                    # Loop over the parameters
                    parameter_values = []
                    for param in parameters:
                        # Randomly select a value for the parameter with truncated normal dsitribution
                        min_val,max_val = self.effects_to_parameters[effect_name][param]
                        value = truncnorm.rvs(min_val,max_val,loc=(min_val+max_val)/2,scale=(max_val-min_val)/4)
                        # Add the parameter to the dictionary
                        effect_data[param] = value
                        parameter_values.append(value)
                    # Create a new effect with the parameters
                    new_effect = effect(**effect_data)
                    # Add the effect to the pedalboard
                    pedalboard.append(new_effect)
                    # Add the effect and corresponding params to the dictionary
                    effect_list.append(Effect(self.effect_to_index[effect_name],parameter_values,len(self.effects),effect_name,self.total_parameters,self.effect_to_param_indices[effect_name],j))
                effect_chain = EffectsChain(effect_list, dry_tone_path, wet_tone_data['wet_tone_path'], len(self.effects),max_chain_length, self.total_parameters)
                wet_tone = pedalboard(dry_tone, sample_rate * f.duration)
                # we don't need to save the actual wet tone because it can be recreated with the dry tone + effect data
                wet_tone_features = feature_extractor(wet_tone, sampling_rate=sample_rate, return_tensors="pt")
                wet_tone_data['wet_tone_features'] = wet_tone_features['input_values'].squeeze(0)
                wet_tone_data['effects'] = effect_chain.effects.squeeze(0)
                wet_tone_data['parameters'] = effect_chain.parameters.squeeze(0)
                wet_tone_data['names'] = effect_chain.names
                wet_tone_data['dry_tone_features'] = dry_tone_features['input_values'].squeeze(0)
                # Append the data to the list
                data.append(wet_tone_data)
        # Return the data
        dataset = EffectChainDataset(data)
        return dataset
