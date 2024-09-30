import pandas as pd
from dataset.data_generate import DataGenerator
from pedalboard import Pedalboard, Chorus, Reverb

df = pd.read_csv('data/neural_chain_dataset/Metadata_Train.csv')
guitar_df = df[df['Class'] == 'Sound_Guitar']
sample_tones = guitar_df['FileName'].tolist()[:20]
effects_parameters = {
    "Chorus": {
        "rate_hz": (0.1, 5.0),
        "depth": (0, 1),
        "centre_delay_ms": (0, 50),
        "feedback": (-1, 1),
        "mix": (0, 1)
    },
    "Reverb": {
        "room_size": (0, 1),
        "damping": (0, 1),
        "wet_level": (0, 1),
        "dry_level": (0, 1),
        "width": (0, 1),
        "freeze_mode": (0, 1)
    }
    }
effects = [Chorus, Reverb]

generator = DataGenerator(effects_parameters, effects)
dataset = generator.create_data(500,'data/neural_chain_dataset/Train_submission/Train_submission',sample_tones,1)