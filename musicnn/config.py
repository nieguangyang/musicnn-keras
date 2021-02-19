# audio signal processing configuration
SR = 16000  # sampling rate
N_FFT = 512  # FFT (fast fourier transform) window size
HOP_LENGTH = 256  # number of frames (i.e. time steps) between STFT columns
N_MELS = 96  # number of frequency bins in mel scale

# model configuration
N_CLASSES = 50
CLIP_LENGTH = 3.  # clip length in seconds
N_TIMESTEPS = int(CLIP_LENGTH * SR // HOP_LENGTH)  # number of time steps per clip
FILTERS = (51, 64, 200)  # numbers of filters for frontend, midend and backend stages of model
