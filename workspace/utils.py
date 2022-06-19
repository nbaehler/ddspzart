import scipy.io.wavfile as wave
import numpy as np

def combine_wavs(wav_paths, out_path):
    """
    Combine multiple wav files into one.
    """
    wavs_data = [wave.read(path)[1] for path in wav_paths]
    max_len = max(data.size for data in wavs_data)
    wavs_data = [np.pad(data, (0, max_len - data.size), 'constant') for data in wavs_data]
    wav = sum(wavs_data).astype(np.int16)
    wave.write(out_path, wav, 44100)
