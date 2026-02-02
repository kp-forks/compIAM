import os
import sys
import torch
import numpy as np
import madmom
from typing import Dict
from tqdm import tqdm
from compiam.exceptions import ModelNotTrainedError

from compiam.rhythm.meter.tcn_carnatic.model import MultiTracker
from compiam.rhythm.meter.tcn_carnatic.pre import PreProcessor
from compiam.rhythm.meter.tcn_carnatic.post import beat_tracker, joint_tracker, sequential_tracker
from compiam.utils.download import download_remote_model
from compiam.utils import get_logger, WORKDIR

logger = get_logger(__name__)

class TCNTracker:
    def __init__(self, post_processor="joint", model_version=42, model_path=None, download_link=None, download_checksum=None, gpu=-1):
        ### IMPORTING OPTIONAL DEPENDENCIES
        try:
            global torch
            import torch

            global MultiTracker
            from compiam.rhythm.meter.tcn_carnatic.model import MultiTracker

        except:
            raise ImportError(
                "In order to use this tool you need to have torch installed. "
                "Install compIAM with torch support: pip install 'compiam[torch]'"
            )
        ###
        if post_processor not in ["beat", "joint", "sequential"]:
            raise ValueError(f"Invalid post_processor: {post_processor}. Choose from 'joint', or 'sequential'.")
        if model_version not in [42, 52, 62]:
            raise ValueError(f"Invalid model_version: {model_version}. Choose from 42, 52, or 62.")

        self.gpu = gpu
        self.device = None
        self.select_gpu(gpu)

        self.model_path = model_path
        self.model_version = f'multitracker_{model_version}.pth'
        self.download_link = download_link
        self.download_checksum = download_checksum

        self.trained = False
        self.model = self._build_model()
        if self.model_path is not None:
            self.load_model(self.model_path)
        self.pre_processor = PreProcessor(fps=100)
        self.pad_frames = 2

        self.post_processor = joint_tracker if post_processor == "joint" else \
                              sequential_tracker

    @torch.no_grad()
    def predict(self, audio_path: str) -> Dict:

        if self.trained is False:
            raise ModelNotTrainedError(
                """Model is not trained. Please load model before running inference!
                You can load the pre-trained instance with the load_model wrapper."""
            )

        features = self.preprocess_audio(audio_path)
        x = torch.from_numpy(features).to(self.device)
        output = self.model(x)
        beats_act = output["beats"].squeeze().detach().cpu().numpy()
        downbeats_act = output["downbeats"].squeeze().detach().cpu().numpy()

        if self.post_processor == beat_tracker:
            beats, beat_positions = self.post_processor(beats_act)
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

    def _build_model(self):
        model = MultiTracker().to(self.device)
        model.eval()
        return model

    def load_model(self, model_path):
        """Load pre-trained model weights."""
        if not os.path.exists(os.path.join(model_path, self.model_version)):
            self.download_model(model_path)  # Downloading model weights

        self.model.load_weights(os.path.join(model_path, self.model_version), self.device)

        self.model_path = model_path
        self.trained = True

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

    def select_gpu(self, gpu="-1"):
        """Select the GPU to use for inference.

        :param gpu: Id of the available GPU to use (-1 by default, to run on CPU), use string: '0', '1', etc.
        :returns: None
        """
        if int(gpu) == -1:
            self.device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:" + str(gpu))
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps:" + str(gpu))
            else:
                self.device = torch.device("cpu")
                logger.warning("No GPU available. Running on CPU.")
        self.gpu = gpu
