#import madmom
from madmom.features.beats import DBNBeatTrackingProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor, DBNBarTrackingProcessor
from scipy.ndimage import maximum_filter1d
import numpy as np

def clip_probabilities(probs):
    """Clip probabilities to avoid exact 0 and 1 values that cause DBN issues."""
    epsilon = 1e-5
    probs = np.maximum(probs, 0)
    probs = np.minimum(probs, 1)
    return probs * (1 - epsilon) + epsilon / 2

def beat_tracker(beats_act, min_bpm=55, max_bpm=230):
    beats_act = clip_probabilities(beats_act)
    beat_dbn = DBNBeatTrackingProcessor(
        min_bpm=min_bpm, max_bpm=max_bpm, fps=100, transition_lambda=100, online=False)

    if beats_act.size > 1:
        beats_pred = beat_dbn(beats_act)
        return beats_pred
    else:
        # If no beats are detected, return an empty array
        return np.array([])

def joint_tracker(beats_act, downbeats_act, min_bpm=55, max_bpm=230):
    beats_act = clip_probabilities(beats_act)
    downbeats_act = clip_probabilities(downbeats_act)

    downbeat_tracker = DBNDownBeatTrackingProcessor(
                        beats_per_bar=[3, 5, 7, 8], min_bpm=min_bpm, max_bpm=max_bpm, fps=100)

    combined_act = np.vstack((np.maximum(beats_act - downbeats_act, 0), downbeats_act)).T
    pred = downbeat_tracker(combined_act)

    return pred

def sequential_tracker(beats_act, downbeats_act, min_bpm=55, max_bpm=230):
    fps = 100
    beats_act = clip_probabilities(beats_act)
    downbeats_act = clip_probabilities(downbeats_act)

    beats = beat_tracker(beats_act)

    # bars (i.e. track beats and then downbeats)
    beat_idx = (beats * fps).astype(int)
    bar_act = maximum_filter1d(downbeats_act, size=3)
    bar_act = bar_act[beat_idx]
    bar_act = np.vstack((beats, bar_act)).T

    bar_tracker = DBNBarTrackingProcessor(beats_per_bar=(3, 5, 7, 8), meter_change_prob=1e-3, observation_weight=4)

    try:
        pred = bar_tracker(bar_act, min_bpm=min_bpm, max_bpm=max_bpm)
    except IndexError:
        pred = np.empty((0, 2))

    return pred
