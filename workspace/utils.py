import scipy.io.wavfile as wave
import numpy as np
from pydub import AudioSegment


def combine_wavs(wav_paths, out_path):
    """
    Combine multiple wav files into one.
    """
    # wavs_data = [wave.read(path)[1] for path in wav_paths]
    # max_len = max(data.size for data in wavs_data)
    # wavs_data = [np.pad(data, (0, max_len - data.size), 'constant') for data in wavs_data]
    # wav = sum(wavs_data).astype(np.int16)
    # wave.write(out_path, 16000, wav) #TODO remove

    wavs = [AudioSegment.from_wav(path) for path in wav_paths]
    result = AudioSegment.empty()

    for wav in wavs:
        result = result.overlay(wav)
    AudioSegment.export(result, out_path, format="wav")

def convert_wav(wav_path, out_path):
    """
    Change the encoding of a wav file to 16-bit.
    """
    rate, data = wave.read(wav_path) # The data is normalized hence in the range [-1, 1]
    data = data + 1
    data /= 2
    data *= 2 ** 16 - 1
    data -= (2 ** 15)

    wave.write(out_path, rate, data.astype(np.int16))
