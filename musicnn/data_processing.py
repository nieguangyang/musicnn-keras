import librosa
import numpy as np

from musicnn.config import SR, N_FFT, HOP_LENGTH, N_MELS


def audio2x(audio_file, clip_length=3, overlap=0, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """
    convert audio file to x, i.e. input to musicnn model
    :param audio_file: str, path to audio file to convert
    :param clip_length: float, clip length in seconds
    :param overlap: float, overlap in seconds
    :param sr: int, sampling rate
    :param n_fft: int, FFT (fast fourier transform) window size
    :param hop_length: int, number of frames (time steps) between STFT columns
    :param n_mels: int, number of frequency bins of mel-spectrogram
    :return x: (batch_size, n_timesteps, n_mels, 1) ndarray, input to musicnn model
    """
    # compute the log-mel spectrogram with librosa
    wav, sr = librosa.load(audio_file, sr=sr)
    spec = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    spec = spec.T
    spec = spec.astype(np.float16)
    spec = np.log10(10000 * spec + 1)
    # batch it for an efficient computing
    n_timesteps = int(clip_length * sr // hop_length)  # window in time steps
    stride = int((clip_length - overlap) * sr // hop_length)  # stride in time steps
    last = spec.shape[0] + 1
    start, end = 0, n_timesteps
    clips = []
    while end < last:
        clip = np.expand_dims(spec[start:end], axis=0)
        clips.append(clip)
        start += stride
        end += stride
    x = np.concatenate(clips, axis=0)
    x = np.expand_dims(x, axis=-1)
    return x


def test():
    from musicnn.example import EXAMPLE
    x = audio2x(EXAMPLE)
    print(x.shape)


if __name__ == "__main__":
    test()
