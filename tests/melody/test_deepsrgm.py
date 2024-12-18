import os
import pytest
import librosa

import numpy as np

from compiam import load_model
from compiam.data import TESTDIR


def _load_model():
    from compiam.melody.raga_recognition import DEEPSRGM

    deepsrgm = DEEPSRGM(
        mapping_path=os.path.join(TESTDIR, "resources", "melody", "raga_mapping.json")
    )
    raga_mapping = deepsrgm.mapping
    assert raga_mapping == {
        0: "Bhairav",
        1: "Madhukauns",
        2: "Mōhanaṁ",
        3: "Hamsadhvāni",
        4: "Varāḷi",
        5: "Dēś",
        6: "Kamās",
        7: "Yaman kalyāṇ",
        8: "Bilahari",
        9: "Ahira bhairav",
    }


def _get_features():
    from compiam.melody.raga_recognition import DEEPSRGM

    deepsrgm = DEEPSRGM(
        mapping_path=os.path.join(TESTDIR, "resources", "melody", "raga_mapping.json")
    )
    raga_mapping = deepsrgm.mapping
    with pytest.raises(FileNotFoundError):
        feat = deepsrgm.get_features(
            os.path.join(TESTDIR, "resources", "melody", "hola.wav")
        )
    with pytest.raises(ValueError):
        feat = deepsrgm.get_features(
            os.path.join(TESTDIR, "resources", "melody", "pitch_test.wav")
        )
    audio = librosa.load(os.path.join(TESTDIR, "resources", "melody", "pitch_test.wav"), sr=44100)[0]
    audio = np.tile(audio, 9)
    feat_1 = deepsrgm.get_features(audio)
    feat_2 = deepsrgm.get_features(np.stack([audio, audio]))
    feat_3 = deepsrgm.get_features(np.stack([audio, audio]).T)


@pytest.mark.torch
def test_predict_tf():
    _load_model()


@pytest.mark.full_ml
def test_predict_full():
    _load_model()


@pytest.mark.all
def test_predict_all():
    _load_model()
    _get_features()
