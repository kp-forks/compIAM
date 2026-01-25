import madmom

from madmom.processors import SequentialProcessor
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor
from scipy.ndimage import maximum_filter1d

from torch.utils.data import Dataset

import numpy as np

# Our approach operates on a spectrogram representation of the audio signal.
# We define the `BeatData` and `PreProcessor` classes which will pre-process our data and transforms the raw audio into a spectrogram with 100 frames per second and 81 frequency bins.

FPS = 100
NUM_BANDS = 12
FFT_SIZE = 2048
MASK_VALUE = -1

# BeatData is a dataset class that handles the loading and processing of beat data from a given dataset.
# It extracts audio, beats, and downbeats, and applies pre-processing to the audio signal
class BeatData(Dataset):
    def __init__(self, dataset, split_keys, fps=100, widen=False):
        self.fps = fps
        self.keys = split_keys
        self.tracks = self._get_tracks(dataset)
        self.pre_processor = PreProcessor(fps=self.fps)
        self.pad_frames = 2
        self.widen = widen

    def _get_tracks(self, dataset):
        tracks = {}
        for k in self.keys:
            tracks[k] = dataset[k]

        return tracks


    def __getitem__(self, idx):
        data = {}
        tid = self.keys[idx]
        track = self.tracks[tid]
        audio, sr = track.audio
        if audio.shape[0] == 2:
            audio = audio.mean(axis=0)

        s = madmom.audio.Signal(audio, sr, num_channels=1)
        x = self.pre_processor(s)

        # pad features
        pad_start = np.repeat(x[:1], self.pad_frames, axis=0)
        pad_stop = np.repeat(x[-1:], self.pad_frames, axis=0)

        x_padded = np.concatenate((pad_start, x, pad_stop))

        if track.beats is None:
            print(f"Warning: Track {self.keys[idx]} has no beat information. Skipping.")
            return self.__getitem__((idx + 1) % len(self))  # Try the next track


        # Extract beat info
        beat_times = track.beats.times.astype(np.float32)
        beat_pos = track.beats.positions.astype(int)
        downbeat_times = beat_times[beat_pos == 1]

        # Quantize beats and downbeats
        beats = madmom.utils.quantize_events(beat_times, fps=self.fps, length=len(x)).astype(np.float32)
        downbeats = madmom.utils.quantize_events(downbeat_times, fps=self.fps, length=len(x)).astype(np.float32)

        if self.widen:
            # we skip masked values (assumed to be -1)
            if not np.allclose(beats, -1):
                np.maximum(beats, maximum_filter1d(beats, size=3) * 0.5, out=beats)
                np.maximum(downbeats, maximum_filter1d(downbeats, size=3) * 0.5, out=downbeats)


        # Adding this because torch is bothered by our batchsize=1
        data["x"] = np.expand_dims(x_padded, axis=0)
        data["key"] = tid
        data["audio"] = audio
        data["sr"] = sr
        data["beats"] = beats
        data["downbeats"] = downbeats
        data["beats_ann"] = beat_times
        data["downbeats_ann"] = downbeat_times

        return data


    def __len__(self):
        return len(self.keys)


# MultiBeatData is a dataset class that handles multiple datasets and their tracks.
# It allows us to work with a list of (dataset_name, key) tuples, making it easier to manage tracks from different datasets.
class MultiBeatData(Dataset):
    def __init__(self, datasets_tracks, split_keys, fps=100, widen=False):
        """
        datasets_tracks: dict of {dataset_name: tracks_dict}
        split_keys: list of (dataset_name, key) tuples
        """
        self.fps = fps
        self.keys = split_keys  # List of (dataset_name, key) tuples
        self.datasets_tracks = datasets_tracks
        self.pre_processor = PreProcessor(fps=self.fps)
        self.pad_frames = 2
        self.widen = widen

    def __getitem__(self, idx):
        data = {}
        dataset_name, key = self.keys[idx]  # Unpack the tuple
        track = self.datasets_tracks[dataset_name][key]  # Get track from correct dataset

        audio, sr = track.audio
        if audio.shape[0] == 2:
            audio = audio.mean(axis=0)

        s = madmom.audio.Signal(audio, sr, num_channels=1)
        x = self.pre_processor(s)

        # pad features
        pad_start = np.repeat(x[:1], self.pad_frames, axis=0)
        pad_stop = np.repeat(x[-1:], self.pad_frames, axis=0)

        x_padded = np.concatenate((pad_start, x, pad_stop))

        if track.beats is None:
            print(f"Warning: Track {dataset_name}:{key} has no beat information. Skipping.")
            return self.__getitem__((idx + 1) % len(self))  # Try the next track

        # Extract beat info
        beat_times = track.beats.times.astype(np.float32)
        beat_pos = track.beats.positions.astype(int)
        downbeat_times = beat_times[beat_pos == 1]

        # Quantize beats and downbeats
        beats = madmom.utils.quantize_events(beat_times, fps=self.fps, length=len(x)).astype(np.float32)
        downbeats = madmom.utils.quantize_events(downbeat_times, fps=self.fps, length=len(x)).astype(np.float32)

        if self.widen:
            # we skip masked values (assumed to be -1)
            if not np.allclose(beats, -1):
                np.maximum(beats, maximum_filter1d(beats, size=3) * 0.5, out=beats)
                np.maximum(downbeats, maximum_filter1d(downbeats, size=3) * 0.5, out=downbeats)

        # Adding this because torch is bothered by our batchsize=1
        data["x"] = np.expand_dims(x_padded, axis=0)
        data["key"] = f"{dataset_name}:{key}"  # Include dataset name for debugging
        data["audio"] = audio
        data["sr"] = sr
        data["beats"] = beats
        data["downbeats"] = downbeats
        data["beats_ann"] = beat_times
        data["downbeats_ann"] = downbeat_times

        return data

    def __len__(self):
        return len(self.keys)


class PreProcessor(SequentialProcessor):
    def __init__(self, frame_size=FFT_SIZE, num_bands=NUM_BANDS, log=np.log, add=1e-6, fps=FPS):
        # resample to a fixed sample rate in order to get always the same number of filter bins
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        # split audio signal in overlapping frames
        frames = FramedSignalProcessor(frame_size=frame_size, fps=fps)
        # compute STFT
        stft = ShortTimeFourierTransformProcessor()
        # filter the magnitudes
        filt = FilteredSpectrogramProcessor(num_bands=num_bands)
        # scale them logarithmically
        spec = LogarithmicSpectrogramProcessor(log=log, add=add)
        # instantiate a SequentialProcessor
        super(PreProcessor, self).__init__((sig, frames, stft, filt, spec, np.array))
        # safe fps as attribute (needed for quantization of events)
        self.fps = fps
