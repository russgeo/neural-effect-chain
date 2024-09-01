from pedalboard import Pedalboard, Delay, Gain, Chorus, Reverb, Distortion, Compressor, Mix, Phaser, NoiseGate, PitchShift, PeakFilter, LowpassFilter, LowShelfFilter, Limiter, LadderFilter, IIRFilter, HighpassFilter, HighShelfFilter, GSMFullRateCompressor, Convolution, Clipping, Invert
from pedalboard.io import AudioFile
import random
import numpy as np
import pandas as pd
import json

# Create a list of all the effects
effects = [Delay, Gain, Chorus, Reverb, Distortion, Compressor, Phaser, NoiseGate, PitchShift, PeakFilter, LowpassFilter, LowShelfFilter, Limiter, LadderFilter, HighpassFilter, HighShelfFilter, Clipping, Invert]
# Create a mapping of effect names to their parameters and max/min values of said parameters
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
    },
    "Invert": {
    }
    }

# "Convolution": {
#         "impulse_response": (0, 1),
#         "mix": (0, 1)
#     }
# "Mix": {
#         "mix": (0, 1)
#     }
# "GSMFullRateCompressor": {
#         "threshold_db": (-100, 0),
#         "ratio": (1, 20),
#         "attack_ms": (0.1, 100),
#         "release_ms": (10, 1000),
#         "makeup_gain_db": (0, 24)
#     }
# "IIRFilter": {
#         "cutoff_frequency_hz": (20, 20000),
#         "q": (0.1, 10)
#     },

def create_data(num_samples, dry_tone_path):
    # Create an empty list to store the data
    data = []
    with AudioFile(dry_tone_path) as f:
        dry_tone = f.read(f.samplerate)
        # Loop over the number of samples
        for i in range(num_samples):
            wet_tone_data = {}
            wet_tone_data['dry_tone_path'] = dry_tone_path
            wet_tone_data['wet_tone_path'] = f'data/wet_tones/output_{i}.wav'
            # Create a new pedalboard
            pedalboard = Pedalboard()
            # Randomly select a number of effects to add to the pedalboard
            num_effects = random.randint(1, 10)
            # Create a dictionary to the effects used and their parameters
            for j in range(num_effects):
                # Randomly select an effect to add
                effect = random.choice(effects)
                # Get the effect name
                effect_name = effect.__name__
                # Get the effect parameters
                parameters = effects_parameters[effect_name].keys()
                # Loop over the parameters
                params_to_vals = {}
                for param in parameters:
                    # Randomly select a value for the parameter
                    value = random.uniform(effects_parameters[effect_name][param][0], effects_parameters[effect_name][param][1])
                    # Add the parameter to the dictionary
                    params_to_vals[param] = value
                # Create a new effect with the parameters
                new_effect = effect(**params_to_vals)
                # Add the effect to the pedalboard
                pedalboard.append(new_effect)
                # Add the effect and corresponding params to the dictionary
                wet_tone_data[effect_name] = params_to_vals

            wet_tone = pedalboard(dry_tone, f.samplerate)
            # Save the wet tone to a file
            with AudioFile(f'data/wet_tones/output_{i}.wav','w',f.samplerate,f.num_channels) as w:
                w.write(wet_tone)
            # Append the data to the list
            data.append(wet_tone_data)
    # Return the data
    return data


data = create_data(5, 'data/dry_tones/Electric1.wav')
pd.DataFrame.from_records(data).to_csv('data/data.csv', index=False)