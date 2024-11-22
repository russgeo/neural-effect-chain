from pedalboard import Pedalboard, Delay, Gain, Chorus, Reverb, Distortion, Compressor, Mix, Phaser, NoiseGate, PitchShift, PeakFilter, LowpassFilter, LowShelfFilter, Limiter, LadderFilter, IIRFilter, HighpassFilter, HighShelfFilter, GSMFullRateCompressor, Convolution, Clipping, Invert
from pedalboard.io import ReadableAudioFile
import random
import pickle
import tqdm
from transformers import AutoFeatureExtractor
import os
#from dataset.dataset import Effect, EffectsChain, EffectChainDataset
from tf_dataset import TFEffectChainDataset, TFEffectsChain, TFEffect
from feature_extractor import FeatureExtractor
import numpy as np
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
class DataGenerator_tf():
    def __init__(self, 
                 effects_to_parameters: dict,
                 effects: list) -> None:
        '''
        Args:
            effects_to_parameters: dictionary of effect names and their parameters and max/min values
            effects: list of effects to use
        Synthetic data generator class, pass in the effects and their parameters to 
        create a synthetic dataset of randomly applied effects with random parameters
        '''
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
                    feature_extractor=FeatureExtractor(),
                    sample_rate=16000
                    ):
        # Create an empty list to store the data
        """
        Args:
            num_samples: number of samples to create per dry tone
            dry_tone_dir: base directory of dry tones
            dry_tones: list of dry tone filenames
            max_chain_length: maximum length of the effect chain
            feature_extractor: feature extractor to use
            sample_rate: sample rate to resample the audio to

        Returns:
            EffectChainDataset: dataset of effect chains
        """
        data = []
        for dry_tone_path in tqdm.tqdm(dry_tones):
            name,ext = dry_tone_path.split('.')
            dry_tone_path = os.path.join(dry_tone_dir, dry_tone_path)
            with ReadableAudioFile(dry_tone_path) as f:
                re_sampled = f.resampled_to(sample_rate)
                # Add padding or truncate to exact length
                desired_length = int(sample_rate * 4)
                dry_tone = re_sampled.read(desired_length)
                if len(dry_tone[0]) < desired_length:
                    # Pad with zeros if audio is too short
                    dry_tone = np.pad(dry_tone, (0, desired_length - len(dry_tone[0])))
                elif len(dry_tone) > desired_length:
                    # Truncate if audio is too long
                    dry_tone = dry_tone[:desired_length]
                re_sampled.close()
                f.close()
            # Loop over the number of samples
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
                        # Randomly select a value for the parameter
                        value = random.uniform(self.effects_to_parameters[effect_name][param][0], self.effects_to_parameters[effect_name][param][1])
                        # Add the parameter to the dictionary
                        effect_data[param] = value
                        parameter_values.append(value)
                    # Create a new effect with the parameters
                    new_effect = effect(**effect_data)
                    # Add the effect to the pedalboard
                    pedalboard.append(new_effect)
                    # Add the effect and corresponding params to the dictionary
                    effect_list.append(TFEffect(self.effect_to_index[effect_name],parameter_values,len(self.effects),effect_name,self.total_parameters,self.effect_to_param_indices[effect_name],j))
                
                effect_chain = TFEffectsChain(effect_list, dry_tone_path, wet_tone_data['wet_tone_path'], len(self.effects),max_chain_length, self.total_parameters)
                wet_tone = pedalboard(dry_tone, sample_rate * 4) #4 for 4 second audio clips
                # we don't need to save the actual wet tone because it can be recreated with the dry tone + effect data
                with tf.device('/cpu:0'):
                    wet_tone_features = feature_extractor.get_log_mel_spectrogram(wet_tone)
                wet_tone_loudness = feature_extractor.get_loudness(wet_tone)
                wet_tone_f0 = feature_extractor.get_f0_crepe(wet_tone)
                wet_tone_data['wet_tone_features'] = wet_tone_features
                wet_tone_data['wet_tone_loudness'] = wet_tone_loudness
                wet_tone_data['wet_tone_f0'] = wet_tone_f0['f0_hz']
                wet_tone_data['effects'] = effect_chain.effects
                wet_tone_data['parameters'] = effect_chain.parameters
                wet_tone_data['names'] = effect_chain.names
                # Append the data to the list
                data.append(wet_tone_data)
        
        return data