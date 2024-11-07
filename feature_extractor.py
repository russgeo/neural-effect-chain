import ddsp
import ddsp.spectral_ops
from ddsp.spectral_ops import PretrainedCREPE
import tensorflow as tf
import torch
import numpy as np
import crepe
from pedalboard.io import ReadableAudioFile 

CREPE_FRAME_SIZE = 1024

def tf_to_torch(tensor):
    """Convert tensorflow tensor to pytorch tensor."""
    if isinstance(tensor, tf.Tensor):
        return torch.from_numpy(tensor.numpy())
    elif isinstance(tensor, np.ndarray):
        return torch.from_numpy(tensor)
    return tensor



def create_audio_representation(audio, sample_rate=16000,frame_rate=250, device='cuda'):
    """Create combined audio representation and return PyTorch tensors.
    
    Args:
        audio: Audio input as numpy array or torch tensor
        sample_rate: Audio sample rate (default: 16000)
        device: PyTorch device to put tensors on (default: 'cuda')
    
    Returns:
        Dictionary of features as PyTorch tensors on specified device
    """
    # Convert input to tensorflow if needed
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    
    # Convert to tensorflow tensor
    audio_tf = tf.convert_to_tensor(audio)
    print(f"audio shape: {audio_tf.shape}")
    # Compute loudness
    loudness = ddsp.spectral_ops.compute_loudness(
        audio_tf, 
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        n_fft=2048,
        range_db=80.0,
        ref_db=0.0,
        use_tf=True
    )

    crepe_step_size = 1000 / frame_rate  # milliseconds
    hop_size = sample_rate // frame_rate

    audio = ddsp.spectral_ops.pad(audio, CREPE_FRAME_SIZE, hop_size, padding='center')
    audio = np.asarray(audio)
    # Compute f0 (fundamental frequency) using CREPE
    crepe_model = PretrainedCREPE('full')
    f0_hz, confidence = crepe_model.predict_f0_and_confidence(audio)
    
    
    # Convert to PyTorch tensors
    return {
        'audio': tf_to_torch(audio_tf).to(device),
        'loudness':tf_to_torch(loudness).to(device),
        'f0_hz': tf_to_torch(f0_hz).to(device),
        'confidence': tf_to_torch(confidence).to(device)
    }


def get_combined_representation(audio_sample, sample_rate=16000, device='cuda'):
    """Get combined audio representation as PyTorch tensors.
    
    Args:
        audio_sample: Audio input as numpy array or torch tensor
        sample_rate: Audio sample rate (default: 16000)
        device: PyTorch device to put tensors on (default: 'cuda')
    
    Returns:
        Dictionary containing:
            - Individual features (f0, loudness, etc.)
            - Combined representation 'z'
        All as PyTorch tensors on specified device
    """
    # Get individual representations
    features = create_audio_representation(audio_sample, sample_rate,frame_rate=250, device=device)
    
    return features

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
    features = get_combined_representation(audio, device='cuda')
    
    # Print shapes
    print("\nFeature shapes:")
    print(f"f0_hz: {features['f0_hz']}")
    print(f"f0 confidence: {features['confidence']}")
    for k, v in features.items():
        print(f"{k}: {tuple(v.shape)}")