import wave
import librosa
import numpy as np
from pydub import AudioSegment
import soundfile as sf


class Utils():

    def __init__(self) -> None:
        pass

    @staticmethod
    def remove_silence(signal: np.ndarray, top_db: int = 65) -> np.ndarray:
        clips = librosa.effects.split(signal, top_db=top_db)
        wav_data = []
        for c in clips:
            data = signal[c[0]: c[1]]
            wav_data.extend(data)
        return np.array(wav_data)

    @staticmethod
    def save_audio_as_wav_and_mp3(signal: np.ndarray, sr: int = 22_050, save_str: str = "audio"):
        sf.write(save_str+".wav", signal, sr)
        sound = AudioSegment.from_wav(save_str+".wav")
        sound.export(save_str+".mp3", format="mp3", codec="libmp3lame")
