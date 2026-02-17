import os
import pytest
import librosa
import torch
import numpy as np

from compiam.data import TESTDIR
from compiam.exceptions import ModelNotTrainedError

def test_predict_joint():
    from compiam.rhythm.meter.tcn_carnatic import TCNTracker

    tracker = TCNTracker()
    with pytest.raises(ModelNotTrainedError):
        tracker.predict(os.path.join(TESTDIR, "resources", "rhythm", "hola.wav"))
    tracker.trained = True
    with pytest.raises(FileNotFoundError):
        tracker.predict(os.path.join(TESTDIR, "resources", "rhythm", "hola.wav"))

    print(tracker.predict(os.path.join(TESTDIR, "resources", "rhythm", "beat_test.wav")))
    beats = tracker.predict(
        os.path.join(TESTDIR, "resources", "rhythm", "beat_test.wav")
    )

    audio_in, sr = librosa.load(
        os.path.join(TESTDIR, "resources", "rhythm", "beat_test.wav")
    )
    beats_2 = tracker.predict(audio_in, sr)

    assert isinstance(beats, np.ndarray)
    assert isinstance(beats_2, np.ndarray)
    assert beats.shape[1] == 2
    assert beats_2.shape[1] == 2

def test_predict_sequential():
    from compiam.rhythm.meter.tcn_carnatic import TCNTracker

    tracker = TCNTracker(post_processor="sequential")
    with pytest.raises(ModelNotTrainedError):
        tracker.predict(os.path.join(TESTDIR, "resources", "rhythm", "hola.wav"))
    tracker.trained = True
    with pytest.raises(FileNotFoundError):
        tracker.predict(os.path.join(TESTDIR, "resources", "rhythm", "hola.wav"))
    beats = tracker.predict(
        os.path.join(TESTDIR, "resources", "rhythm", "beat_test.wav")
    )

    audio_in, sr = librosa.load(
        os.path.join(TESTDIR, "resources", "rhythm", "beat_test.wav")
    )
    beats_2 = tracker.predict(audio_in)

    assert isinstance(beats, np.ndarray)
    assert isinstance(beats_2, np.ndarray)
    assert beats.shape[1] == 2
    assert beats_2.shape[1] == 2

def test_48k():
    from compiam.rhythm.meter.tcn_carnatic import TCNTracker

    tracker = TCNTracker(post_processor="sequential")
    tracker.trained = True
    audio_in, sr = librosa.load(
        os.path.join(TESTDIR, "resources", "rhythm", "48k.wav"), sr=48000
    )
    beats = tracker.predict(audio_in)

    assert isinstance(beats, np.ndarray)
    assert beats.shape[1] == 2
