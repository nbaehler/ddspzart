from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Iterable, Union, Tuple, Any, Optional, Generic, TypeVar, AnyStr, Iterator
import numpy as np
import logging

try:
    USE_PYRUBBERBAND = True
    import pyrubberband
except:
    logging.warning("pyrubberband cannot be importe available. Use custom implementations.")
    USE_PYRUBBERBAND = False



from scipy import interpolate

@dataclass
class AugmentationConfig:
    AUDIO_SAMPLE_RATE: int = 16e3

class Utils:
    
    def __init__(self) -> None:
        pass
        
    @staticmethod#
    def ms2n(ms, sf):
        return int(float(sf) * float(ms) / 1000.0)
    
    @staticmethod
    def resample(x, alpha):
        # length of the output signal after resampling
        n_out = int(np.floor(len(x) * alpha))
        y = np.zeros(n_out)
        for iy in range(0, n_out - 1):
            t = iy / alpha 
            ix = int(t)
            y[iy] = (1 - (t - ix)) * x[ix] + (t - ix) * x[ix + 1] 
        return y

    @staticmethod
    def tapering_window(N, overlap):
        R = int(N * overlap / 2)
        r = np.arange(0, R) / float(R)
        win = np.r_[r, np.ones(N - 2*R), r[::-1]]
        stride = N - R - 1 if R > 0 else N
        return win, stride

    @staticmethod
    def hanning_window(size):
        # make sure size is odd
        stride = size // 2
        size = 2 * stride + 1
        win = np.hanning(size)
        return win, stride, size


TInput = TypeVar('TInput')
TOutput = TypeVar('TOutput')
class Transformer(Generic[TInput, TOutput]):
    def __init__(self, name: str, save_as_audio: bool = False):
        self.name = name
        self.save_as_audio = save_as_audio

    def __call__(self, value: TInput) -> TOutput:
        return self.transform(value)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={repr(self.name)})'

    @abstractmethod
    def transform(self, input: TInput) -> TOutput:
        raise NotImplementedError('This abstract method has not been implemented.')

    def transform_multi(self, inputs: Iterable[TInput]) -> Iterator[TOutput]:
        yield from (self(value) for value in inputs)

class PitchShiftTransformer(Transformer[np.ndarray, np.ndarray]):

    def __init__(self, sample_rate: Optional[int] = None, shift_semitones: int = 0):
        self.sample_rate = sample_rate if not sample_rate is None else AugmentationConfig.AUDIO_SAMPLE_RATE
        self.sample_rate = int(self.sample_rate)
        self.shift_semitones = shift_semitones
        super().__init__(f'pitch_shift-({self.sample_rate}-{self.shift_semitones})')

    def transform(self, input: np.ndarray) -> np.ndarray:
        """input is expected to be a mono channel of samples."""
        #return self.transform_with_com418(input) #FIXME
        if not len(input.shape) == 1:
            raise ValueError(f'Expected input data with a dimensionality of 1, got {len(input.shape)} (exact shape was {repr(input.shape)}).')
        return self.pitch_shift(input)

    def pitch_shift(self, input: np.ndarray) -> np.ndarray:
        if USE_PYRUBBERBAND:
            return pyrubberband.pyrb.pitch_shift(input, self.sample_rate, self.shift_semitones)
        else:
            logging.warning("pyrubberband not available. Use custom implementation.")
            return self.transform_with_com418(input) 
        

    def pitchshift_gs_rt(self, x, alpha, grain_size, overlap=0.4):
        win, stride = Utils.tapering_window(grain_size, overlap)
        # resampling needs these many input samples to produce an output grain of the chosen size
        chunk_size = int(np.floor(grain_size + 1) * alpha)
        y = np.zeros(len(x))
        # input chunks and output grains are always aligned in pitch shifting (in_hop = out_hop = stride)
        for n in range(0, len(x) - max(chunk_size, grain_size), stride):
            
            try:
                y[n:n+grain_size] += Utils.resample(x[n:n+chunk_size], 1 / alpha) * win
            except: #FIXME Check why this happens
                print('warning')
                y[n:n+grain_size-1] += Utils.resample(x[n:n+chunk_size], 1 / alpha) * win

        return y

    def transform_with_com418(self, input: np.ndarray) -> np.ndarray:
        semitone = 2 ** (1.0 / 12)
        grain_size = Utils.ms2n(100, self.sample_rate)
        return self.pitchshift_gs_rt(input, semitone ** (self.shift_semitones), grain_size)

class PitchShiftAugmentationTransformer(PitchShiftTransformer):
    def __init__(self, sample_rate: Optional[int] = None, max_shift: int = 0):
        super().__init__(sample_rate, max_shift)
        self.max_shift = max_shift

    def transform(self, input: np.ndarray) -> np.ndarray:
        """input is expected to be a mono channel of samples."""
        self.shift_semitones = int(np.random.uniform(-self.max_shift, self.max_shift))
        return self.pitch_shift(input)


class TimeDownsamplingTransformer(Transformer[np.ndarray, np.ndarray]):
    def __init__(self, downsampling_factor: float = 1.0, sample_rate: int = None):
        self.downsampling_factor = downsampling_factor
        self.sample_rate = sample_rate if sample_rate is not None else AugmentationConfig.AUDIO_SAMPLE_RATE
        super().__init__(f'spectrogram_time_downsampling_{downsampling_factor}')

    def transform(self, input: np.ndarray) -> np.ndarray:
        """Input is expected to be a spectrogram (e.g. generated by SciPyAbsoluteSTFTTransformer or AbsoluteConstantQTransformer) with two dimensions where time is the last."""
        if len(input.shape) == 2:
            if USE_PYRUBBERBAND:
                return pyrubberband.pyrb.time_stretch(input, self.samplerate, self.strectch_factor)
            else:
                logging.warning("pyrubberband not available. Use custom implementation.")
                bins, times = input.shape
                times_downsampled = max(1, int(times / self.downsampling_factor))
                xrange = lambda x: np.linspace(0, 1, x)
                return interpolate.interp2d(xrange(times), xrange(bins), input, kind='linear')(xrange(times_downsampled), xrange(bins))
        else:
            raise ValueError(f'Expected input data with a dimensionality of 2, got {len(input.shape)} (exact shape was {repr(input.shape)}).')
    
class TimeStretchingTransformer(Transformer[np.ndarray, np.ndarray]):
    def __init__(self, stretch_factor: float = 1.0, sample_rate: int = None):
        self.stretch_factor = stretch_factor
        self.sample_rate = sample_rate if sample_rate is not None else AugmentationConfig.AUDIO_SAMPLE_RATE
        super().__init__(f'time_stretching_{stretch_factor}')

    def transform(self, input: np.ndarray) -> np.ndarray:
        return self.time_stretching(input)
    
    def time_stretching(self, input: np.ndarray) -> np.ndarray:
        if USE_PYRUBBERBAND:
            return pyrubberband.pyrb.time_stretch(input, self.sample_rate, self.stretch_factor)
        else:
            logging.warning("pyrubberband not available. Use custom implementation.")
            return self.transform_com418(input)


    def transform_com418(self,input: np.ndarray) -> np.ndarray:
        grain_size = Utils.ms2n(100, self.sample_rate)
        return self.timescale_gs_pv(input, self.strectch_factor, grain_size)

    def timescale_gs_pv(self,x, alpha, grain_size):
        # we will use an odd-length Hanning window with 100% overlap
        win, stride, grain_size = Utils.hanning_window(grain_size)
        in_hop, out_hop = int(stride / alpha), stride
        # initialize output phase with phase of first grain
        phase = np.angle(np.fft.fft(x[0:grain_size]))
        y, ix, iy = np.zeros(int(alpha * len(x))), 0, 0    
        while ix < len(x) - 2 * grain_size and iy < len(y) - grain_size:
            # FFT of current grain
            grain_fft = np.fft.fft(win * x[ix:ix+grain_size])
            # phase of the grain at the point of intersection with next grain in the output 
            end_phase = np.angle(np.fft.fft(win * x[ix+out_hop:ix+out_hop+grain_size]))
            phase_diff = end_phase - np.angle(grain_fft)
            # compute rephased grain and add with overlap to output
            grain = np.real(np.fft.ifft(np.abs(grain_fft) * np.exp(1j * phase)))
            y[iy:iy+grain_size] += grain * win
            iy += out_hop
            ix += in_hop
            # update output phase for next grain and reduce modulo 2pi
            phase = phase + phase_diff  
            phase = phase - 2 * np.pi * np.round(phase / (2 * np.pi))
        return y

class TimeStretchingAugmentationTransformer(TimeStretchingTransformer):

    def __init__(self, min_stretch_factor: float = 1, max_stretch_factor: float = 1, sample_rate: int = None):
        super().__init__(max_stretch_factor, sample_rate)
        self.max_stretch_factor = max_stretch_factor
        self.min_stretch_factor = min_stretch_factor

    def transform(self, input: np.ndarray) -> np.ndarray:
        self.stretch_factor = np.random.uniform(self.min_stretch_factor, self.max_stretch_factor)
        return self.time_stretching(input)
  

class LoudnessTransformer(Transformer[np.ndarray, np.ndarray]):
    
    def __init__(self, f: float = 1):
        super().__init__(f'loudness_{f}')
        self.f = f

    def transform(self, input: np.ndarray) -> np.ndarray:
        return input * self.f


class LoudnessAugmentationTransformer(Transformer[np.ndarray, np.ndarray]):
    def __init__(self, f_max: float = 1):
        super().__init__(f'loudness_augmentation_{f_max}')
        self.f_max = f_max

    def transform(self, input: np.ndarray) -> np.ndarray:
        f = np.random.uniform(1 / self.f_max, self.f_max)
        return input * f
    
class DiverseLoudnessTransformer(Transformer[np.ndarray, np.ndarray]):

    def __init__(self, f_max: float = 1, Tc: float = 1.0, sample_rate: int = None):
        super().__init__(f'diverse_loudness_{f_max}')
        self.f_max = f_max
        self.sample_rate = sample_rate if sample_rate is not None else AugmentationConfig.AUDIO_SAMPLE_RATE
        self.N = int(Tc * self.sample_rate)

    def get_factor(self) -> np.uint:
        return self.f_max

    def transform(self, x: np.ndarray) -> np.ndarray:
        
        self.f = self.get_factor()
        weigthing = np.random.randint(0,2,len(x))
        N = self.N
        weigthing = 2*np.convolve(weigthing, np.ones(N)/N, mode='valid') - 1 
        weigthing = np.concatenate((weigthing[0:(len(x)-len(weigthing))//2],weigthing,weigthing[-(len(x)-len(weigthing))//2:]))
        weigthing = weigthing - np.mean(weigthing)
        weigthing -= np.min(weigthing)
        weigthing *= self.f / np.max(weigthing)

        return x * weigthing

class DiverseLoudnessAugmentationTransformer(DiverseLoudnessTransformer):

        def __init__(self, f_max: float = 1, Tc: float = 1.0, sample_rate: int = None):
            super().__init__(f_max, Tc, sample_rate)

        def get_factor(self) -> np.uint:
            return np.random.uniform(1 / self.f_max, self.f_max)

class AdditiveGaussianNoiseTransformer(Transformer[np.ndarray, np.ndarray]):
    def __init__(self, mean: float = 0, sigma: float = 0):
        super().__init__(f'additive_gaussian_noise_{mean}_{sigma}')
        self.mean = mean
        self.sigma = sigma

    def transform(self, input: np.ndarray) -> np.ndarray:
        noise_matrix = np.random.normal(self.mean, self.sigma, input.shape)
        return input + noise_matrix

class AdditiveGaussianNoiseAugmentationTransformer(Transformer[np.ndarray, np.ndarray]):
    def __init__(self, mean: float = 0, sigma_max: float = 0):
        super().__init__(f'additive_gaussian_noise_augmentation_{mean}_{sigma_max}')
        self.mean = mean
        self.sigma_max = sigma_max

    def transform(self, input: np.ndarray) -> np.ndarray:
        sigma = np.random.uniform(0, self.sigma_max)
        noise_matrix = np.random.normal(self.mean, sigma, input.shape)
        return input + noise_matrix

# # This is only for spectrogram as input
# class TimeFrequencyDomainTimeWarpTransformer(Transformer[np.ndarray, np.ndarray]):
#     def __init__(self, warp_distance: float = 0):
#         super().__init__(f't_f_time_warp_{warp_distance}')
#         self.warp_distance = warp_distance

#     def transform(self, input: np.ndarray) -> np.ndarray:
#         bins, times = input.shape
#         xrange = lambda x: np.linspace(0, 1, x)
#         x_1 = times // 2 + self.warp_distance
#         x_2 = times - x_1
#         xwarprange = np.concatenate((np.linspace(0, .5, x_1, endpoint=False), np.linspace(.5, 1, x_2)))
#         return interpolate.interp2d(xrange(times), xrange(bins), input, kind='linear')(xwarprange, xrange(bins))

# class TimeFrequencyDomainTimeWarpAugmentationTransformer(Transformer[np.ndarray, np.ndarray]):
#     def __init__(self, max_warp_distance: float = 0):
#         super().__init__(f't_f_time_warp_augmentation_{max_warp_distance}')
#         self.max_warp_distance = max_warp_distance

#     def transform(self, input: np.ndarray) -> np.ndarray:
#         warp_distance = int(np.random.uniform(-self.max_warp_distance, self.max_warp_distance))
#         bins, times = input.shape
#         xrange = lambda x: np.linspace(0, 1, x)
#         x_1 = times // 2 + warp_distance
#         x_2 = times - x_1
#         xwarprange = np.concatenate((np.linspace(0, .5, x_1, endpoint=False), np.linspace(.5, 1, x_2)))
#         return interpolate.interp2d(xrange(times), xrange(bins), input, kind='linear')(xwarprange, xrange(bins))

# # Not really useful
# class TimeMaskingTransformer(Transformer[np.ndarray, np.ndarray]):
#     def __init__(self, t_0: float, t: float):
#         super().__init__(f'time_mask_{t_0}_{t}')
#         self.t_0 = t_0
#         self.t = t

#     def transform(self, input: np.ndarray) -> np.ndarray:
#         input[:,self.t_0:(self.t_0+self.t)] = 0
#         return input

# # Not really useful
# class TimeMaskingAugmentationTransformer(Transformer[np.ndarray, np.ndarray]):
#     def __init__(self, max_t: float):
#         super().__init__(f'time_mask_augmentation_{max_t}')
#         self.max_t = max_t

#     def transform(self, input: np.ndarray) -> np.ndarray:
#         t = int(np.random.uniform(0, self.max_t))
#         t_0 = int(np.random.uniform(0, input.shape[1] - t - 1))
#         input[:,t_0:(t_0+t)] = 0
#         return input

# # Not really useful
# class FrequencyMaskingTransformer(Transformer[np.ndarray, np.ndarray]):
#     def __init__(self, f_0: float, f: float):
#         super().__init__(f'frequency_mask_{f_0}_{f}')
#         self.f_0 = f_0
#         self.f = f

#     def transform(self, input: np.ndarray) -> np.ndarray:
#         input[self.f_0:(self.f_0+self.f)] = 0
#         return input


# # Not really useful
# class FrequencyMaskingAugmentationTransformer(Transformer[np.ndarray, np.ndarray]):
#     def __init__(self, max_f: float):
#         super().__init__(f'frequency_mask_augmentation_{max_f}')
#         self.max_f = max_f

#     def transform(self, input: np.ndarray) -> np.ndarray:
#         f = int(np.random.uniform(0, self.max_f))
#         f_0 = int(np.random.uniform(0, input.shape[0] - f - 1))
#         input[f_0:(f_0+f)] = 0
#         return input

# # Not really useful
# class FrequencyMaskingTransformer(Transformer[np.ndarray, np.ndarray]):
#     def __init__(self, f_0: float, f: float):
#         super().__init__(f'frequency_mask_{f_0}_{f}')
#         self.f_0 = f_0
#         self.f = f

#     def transform(self, input: np.ndarray) -> np.ndarray:
#         input[self.f_0:(self.f_0+self.f)] = 0
#         return input