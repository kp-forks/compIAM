import os
import sys
import torch
import numpy as np
import madmom
from typing import Dict

from compiam.rhythm.beat_tracking.tcn_tracker.model import MultiTracker
from compiam.rhythm.beat_tracking.tcn_tracker.pre import PreProcessor
from compiam.rhythm.beat_tracking.tcn_tracker.post import beat_tracker, joint_tracker, sequential_tracker
from compiam.utils.download import download_remote_model
from compiam.utils import get_logger, WORKDIR

logger = get_logger(__name__)

class TCNTracker:
    def __init__(self, post_processor: str = "joint", model_version: int = 42, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if post_processor not in ["beat", "joint", "sequential"]:
            raise ValueError(f"Invalid post_processor: {post_processor}. Choose from 'beat', 'joint', or 'sequential'.")
        if model_version not in [42, 52, 62]:
            raise ValueError(f"Invalid model_version: {model_version}. Choose from 42, 52, or 62.")

        n_filters = 20
        kernel_size = 5
        n_dilations = 11
        self.model_path = model_path
        self.model_version = f'multitracker_{model_version}.pth'
        self.download_link = 'https://zenodo.org/records/18449067/files/compIAM-TCNCarnatic.zip?download=1'
        self.download_checksum = '995369933f2a344af0ffa57ea5c15e62'

        self.model = MultiTracker(n_filters=n_filters, n_dilations=n_dilations, kernel_size=kernel_size).to(self.device)
        self.load_model(self.model_path)
        self.model.eval()
        self.pre_processor = PreProcessor(fps=100)
        self.pad_frames = 2

        self.post_processor = beat_tracker if post_processor == "beat" else \
                              joint_tracker if post_processor == "joint" else \
                              sequential_tracker

    @torch.no_grad()
    def __call__(self, audio_path: str) -> Dict:
        features = self.preprocess_audio(audio_path)
        x = torch.from_numpy(features).to(self.device)
        output = self.model(x)
        beats_act = output["beats"].squeeze().detach().cpu().numpy()
        downbeats_act = output["downbeats"].squeeze().detach().cpu().numpy()

        if self.post_processor == beat_tracker:
            pred = self.post_processor(beats_act)
        else:
            pred = self.post_processor(beats_act, downbeats_act)
        return pred

    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        audio, sr = madmom.io.audio.load_audio_file(audio_path)

        if audio.shape[0] == 2:
            audio = audio.mean(axis=0)

        s = madmom.audio.Signal(audio, sr, num_channels=1)
        x = self.pre_processor(s)

        pad_start = np.repeat(x[:1], self.pad_frames, axis=0)
        pad_stop = np.repeat(x[-1:], self.pad_frames, axis=0)
        x_padded = np.concatenate((pad_start, x, pad_stop))

        x_final = np.expand_dims(np.expand_dims(x_padded, axis=0), axis=0)

        return x_final

    def load_model(self, model_path):
        path_to_check = os.path.join(model_path, self.model_version)
        if not os.path.exists(path_to_check):
            print('yes')
            self.download_model(model_path)  # Downloading model weights
        self.model.load_weights(os.path.join(model_path, self.model_version), self.device)

    def download_model(self, model_path=None, force_overwrite=True):
        """Download pre-trained model."""
        download_path = (
            #os.sep + os.path.join(*model_path.split(os.sep)[:-2])
            model_path
            if model_path is not None
            else os.path.join(WORKDIR, "models", "rhythm", "tcn-carnatic")
        )
        # Creating model folder to store the weights
        if not os.path.exists(download_path):
            os.makedirs(download_path)
        download_remote_model(
            self.download_link,
            self.download_checksum,
            download_path,
            force_overwrite=force_overwrite,
        )
