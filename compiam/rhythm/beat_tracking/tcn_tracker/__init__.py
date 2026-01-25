import os
import sys
import torch
import numpy as np
import madmom
from typing import Dict

from compiam.rhythm.beat_tracking.tcn_tracker.tcn import MultiTracker
from compiam.rhythm.beat_tracking.tcn_tracker.data_loader import PreProcessor

class TCNTracker:
    def __init__(self, checkpoint_path: str, model_params: Dict = None, post_processor: str = "JOINT"):
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Default model parameters (should match training)
        if model_params is None:
            model_params = {
                "N_FILTERS": 20,
                "KERNEL_SIZE": 5,
                "DROPOUT": 0.15,
                "N_DILATIONS": 11,
            }

        params = {
            "LEARNING_RATE": 0.001,
            "LOSS": "BCE",
            "POST_PROCESSOR": post_processor,
            "SCHEDULER_FACTOR": 0.2,
            "SCHEDULER_PATIENCE": 5
        }

        # Create the model architecture (same as in training)
        tcn = MultiTracker(
            n_filters=model_params["N_FILTERS"],
            n_dilations=model_params["N_DILATIONS"],
            kernel_size=model_params["KERNEL_SIZE"],
            dropout_rate=model_params["DROPOUT"]
        )

        self.model.eval()

        # Initialize preprocessor (same as your data loader)
        self.pre_processor = PreProcessor(fps=100)
        self.pad_frames = 2

    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """Extract features using your existing pipeline - EXACTLY like data_loader.py"""
        # Load audio using madmom (same as your data loader)
        audio, sr = madmom.io.audio.load_audio_file(audio_path)

        # Handle stereo to mono (same logic as your data loader)
        if audio.shape[0] == 2:
            audio = audio.mean(axis=0)

        # Create signal and extract features (same as your data loader)
        s = madmom.audio.Signal(audio, sr, num_channels=1)
        x = self.pre_processor(s)

        # Apply padding (same as your data loader)
        pad_start = np.repeat(x[:1], self.pad_frames, axis=0)
        pad_stop = np.repeat(x[-1:], self.pad_frames, axis=0)
        x_padded = np.concatenate((pad_start, x, pad_stop))

        # IMPORTANT: Apply the same dimension expansion as in data_loader.py
        # The model expects (batch, channels, height, width) for 2D convolutions
        # x_padded has shape (time, frequency_bins) where frequency_bins=12
        # We need to reshape it to (batch=1, channels=1, height=time, width=frequency_bins)
        x_final = np.expand_dims(np.expand_dims(x_padded, axis=0), axis=0)

        return x_final

    @torch.no_grad()
    def predict(self, audio_path: str) -> Dict:
        """Predict beats from audio file"""
        # Extract features using your pipeline (now with correct dimensions)
        features = self.preprocess_audio(audio_path)

        # Convert to tensor - features now have shape (1, 1, time, frequency_bins)
        x = torch.from_numpy(features).to(self.device)

        # Forward pass
        output = self.model(x)

        # Extract activations
        beats_act = output["beats"].squeeze().detach().cpu().numpy()
        downbeats_act = output["downbeats"].squeeze().detach().cpu().numpy()

        # Apply post-processing using the model's post_tracker
        pred = self.model.post_tracker(beats_act, downbeats_act)

        return pred
