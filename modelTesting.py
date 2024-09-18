from model.EffectDecoder import EffectDecoder
import numpy as np
import librosa
from transformers import AutoFeatureExtractor

dry_y, dry_sr = librosa.load('data/wet_tones/output_1.wav',sr=16000)

wet_y, wet_sr = librosa.load('data/wet_tones/output_2.wav',sr=16000)

spectrogram_feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

dry_spec = spectrogram_feature_extractor(dry_y,sampling_rate = dry_sr,return_tensors="pt")
wet_spec = spectrogram_feature_extractor(wet_y,sampling_rate = wet_sr,return_tensors="pt")

effect_decoder = EffectDecoder(dry_sr,wet_sr,10)
output = effect_decoder(dry_spec,wet_spec)

print(output)