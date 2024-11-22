from dataset.data_generator import DataGenerator
import pandas as pd
from pedalboard import Pedalboard, Chorus, Reverb, Delay, Gain, Distortion
import pickle
import json
import tensorflow as tf
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
    }
    }

effects = [Chorus, Reverb, Delay, Gain, Distortion]

generator = DataGenerator(effects_parameters, effects)

with open('data/nsynth-train.jsonwav/nsynth-train/examples.json', 'r') as f:
    data = json.load(f)
df = pd.DataFrame.from_records(data)
df = df.T
guitar_df = df[df['instrument_family_str'] == 'guitar']
elctric_guitar_df = guitar_df[guitar_df['instrument_source_str'] == "electronic"][:100]
dry_tones = [dry_tone + ".wav" for dry_tone in elctric_guitar_df['note_str'].tolist()]

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# generate the dataset
import tensorflow as tf

# Assuming dataset is a list of dictionaries
dataset = generator.create_data(num_samples=5, dry_tone_dir='data/nsynth-train.jsonwav/nsynth-train/audio', dry_tones=dry_tones, max_chain_length=1)



# Serialize and Save the Dataset as TFRecord
def serialize_example(entry):
    feature = {
        "wet_tone_features": tf.train.Feature(float_list=tf.train.FloatList(value=entry["wet_tone_features"].numpy().flatten())),
        "effects": tf.train.Feature(float_list=tf.train.FloatList(value=entry["effects"].numpy().flatten())),
        "dry_tone_path": tf.train.Feature(bytes_list=tf.train.BytesList(value=[entry["dry_tone_path"].numpy()])),
        "wet_tone_loudness": tf.train.Feature(float_list=tf.train.FloatList(value=[entry["wet_tone_loudness"].numpy()])),
        "wet_tone_f0": tf.train.Feature(float_list=tf.train.FloatList(value=[entry["wet_tone_f0"].numpy()])),
        "parameters": tf.train.Feature(float_list=tf.train.FloatList(value=[entry["parameters"].numpy()])),
        "names": tf.train.Feature(bytes_list=tf.train.BytesList(value=[entry["names"].numpy()])),
    }
    return tf.train.Example(features=tf.train.Feature(feature=feature)).SerializeToString()

# Save to TFRecord file
file_path = "data/nsynth_dataset_tf.tfrecord"
with tf.io.TFRecordWriter(file_path) as writer:
    for entry in ds:
        writer.write(serialize_example(entry))

print(f"Dataset saved to {file_path}")
