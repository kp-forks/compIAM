import numpy as np
from madmom.processors import SequentialProcessor
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor
from torch.utils.data import Dataset


class PreProcessor(SequentialProcessor):
    def __init__(self, sample_rate, frame_size=2048, num_bands=12, log=np.log, add=1e-6, fps=100):
        sig = SignalProcessor(num_channels=1, sample_rate=sample_rate)
        frames = FramedSignalProcessor(frame_size=frame_size, fps=fps)
        stft = ShortTimeFourierTransformProcessor()
        filt = FilteredSpectrogramProcessor(num_bands=num_bands)
        spec = LogarithmicSpectrogramProcessor(log=log, add=add)
        super(PreProcessor, self).__init__((sig, frames, stft, filt, spec, np.array))
        self.fps = fps
