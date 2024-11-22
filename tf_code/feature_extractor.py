import ddsp
import ddsp.spectral_ops
from ddsp.spectral_ops import PretrainedCREPE, compute_mel, compute_logmel
import tensorflow as tf
import numpy as np
from pedalboard.io import ReadableAudioFile
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

F0_RANGE = 127.0  # MIDI.
DB_RANGE = 80  # dB (80.0).

CREPE_FRAME_SIZE = 1024


class FeatureExtractor():
    def __init__(self, sample_rate=16000,frame_rate=250):
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        return
    
    def get_f0_crepe(self, audio):
        """Create combined audio representation and return PyTorch tensors.
    
    Args:
        audio: Audio input as numpy array or torch tensor
        sample_rate: Audio sample rate (default: 16000)
        device: tensorflow device to put tensors on (default: 'cuda')
    
    Returns:
        Dictionary of features as Tensorflow tensors on specified device
        """
        # Convert input to tensorflow if needed
        # Convert to tensorflow tensor
        audio_tf = tf.convert_to_tensor(audio)
        

        crepe_step_size = 1000 / self.frame_rate  # milliseconds
        hop_size = self.sample_rate // self.frame_rate

        audio = ddsp.spectral_ops.pad(audio, CREPE_FRAME_SIZE, hop_size, padding='center')
        audio = np.asarray(audio)
        # Compute f0 (fundamental frequency) using CREPE
        crepe_model = PretrainedCREPE('full')
        f0_hz, confidence = crepe_model.predict_f0_and_confidence(audio)
        #return features
        return {'f0_hz': f0_hz,'confidence': confidence}
    
    def get_f0_basic_pitch(self, audio_path):
        model_output, midi_data, note_events = predict(audio_path)
        return model_output
    
    def get_loudness(self, audio):
        audio_tf = tf.convert_to_tensor(audio)
        loudness = ddsp.spectral_ops.compute_loudness(
        audio_tf, 
        sample_rate=self.sample_rate,
        frame_rate=self.frame_rate,
        n_fft=2048,
        range_db=80.0,
        ref_db=0.0,
        use_tf=True
        )
        return loudness

    def get_midi(self, audio):
        model_output, midi_data, note_events = predict(audio)
        return midi_data
    
    def get_mel_spectrogram(self, audio):
        audio_tf = tf.convert_to_tensor(audio)
        spectrogram = compute_mel(audio_tf, bins=128,pad_end=True)
        return spectrogram
    
    def get_log_mel_spectrogram(self, audio):
        audio_tf = tf.convert_to_tensor(audio)
        mel_spectrogram = compute_logmel(audio_tf,bins=128)
        return mel_spectrogram
    
    def get_features(self, audio, f_0="crepe", log_spectrogram=True, midi=False):
        '''
        Get a dictionary of features from the audio
        Args:
            audio: Audio input as numpy array or torch tensor
            f_0: "crepe" or "basic_pitch"
            spectrogram: "mel" or "linear"
            midi: True or False
        Returns:
            Dictionary of features
            f0: fundamental frequency/pitch
            loudness: loudness
            midi: midi data (if midi=True)
            spectrogram: spectrogram
        '''
        if f_0 == "crepe":
            f0 = self.get_f0_crepe(audio)
        elif f_0 == "basic_pitch":
            f0 = self.get_f0_basic_pitch(audio)
        loudness = self.get_loudness(audio)
        if log_spectrogram:
            spec = self.get_log_mel_spectrogram(audio)
        else:
            spec = self.get_mel_spectrogram(audio)
        if midi:
            midi_data = self.get_midi(audio)
            
            return {'f0': f0, 'loudness': loudness, 'midi': midi_data, 'spectrogram': spec}
        else:
            return {'f0': f0, 'loudness': loudness, 'spectrogram': spec}


# Example usage
if __name__ == '__main__':
    # Create sample audio (1 second at 16kHz)
    SAMPLE_RATE = 16000
    with ReadableAudioFile("data/dry_tones/Electric1.wav") as f:
        # re sample the audio file to match the sample rate, pretrained model is sampled at 16000
        re_sampled = f.resampled_to(SAMPLE_RATE)
        audio = re_sampled.read(int(SAMPLE_RATE * f.duration))
        re_sampled.close()
        f.close()
    # Trim audio to 4 seconds if longer
    if len(audio) > SAMPLE_RATE * 4:
        audio = audio[:SAMPLE_RATE * 4]
    # Get representations
    feature_extractor = FeatureExtractor()
    f0_crepe = feature_extractor.get_f0_crepe(audio)
    f0_basic_pitch = feature_extractor.get_f0_basic_pitch(audio)
    loudness = feature_extractor.get_loudness(audio)
    midi = feature_extractor.get_midi(audio)

    
    # Print shapes