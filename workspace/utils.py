import scipy.io.wavfile as wave
import numpy as np
from pydub import AudioSegment


def combine_wavs(wav_paths, out_path):
    """
    Combine multiple wav files into one.
    """

    if not wav_paths:
        raise ValueError("List of wav files is empty")

    wavs = [AudioSegment.from_wav(path) for path in wav_paths]
    result = wavs[0]

    for wav in wavs[1:]:
        result = result.overlay(wav)

    AudioSegment.export(result, out_path, format="wav")


def convert_wav(wav_path, out_path):
    """
    Change the encoding of a wav file to 16-bit.
    """
    rate, data = wave.read(
        wav_path)  # The data is normalized hence in the range [-1, 1]
    data = data + 1
    data /= 2
    data *= 2 ** 16 - 1
    data -= (2 ** 15)

    wave.write(out_path, rate, data.astype(np.int16))
