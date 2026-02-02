import compiam
import os



import numpy as np
import matplotlib.pyplot as plt
import librosa
import IPython.display as ipd

#@title Plotting function
def plot_sonify(y, sr=22050, beats=None, labels=None, start_time=0, duration=None):
    """
    Plots the waveform with optional beats and downbeats. If beats are None, only the waveform is plotted.

    Parameters:
    - y: Audio signal waveform data.
    - sr: Sampling rate of the audio signal.
    - beats: Array of beat times (in seconds). If None, only the waveform is plotted.
    - labels: Optional array of labels corresponding to beats. Downbeats should be labeled as 1.
    - start_time: Start time in seconds for the plot window (default: 0).
    - duration: Duration in seconds for the plot window. If None, plots entire signal.
    """
    time = (np.arange(len(y))/sr)

    plt.figure(figsize=(100, 5))

    # Apply time windowing if specified
    plot_y = y
    plot_time = time

    if duration is not None:
        end_time = start_time + duration
        # Find indices for the time window
        start_idx = np.searchsorted(time, start_time)
        end_idx = np.searchsorted(time, end_time)
        plot_time = time[start_idx:end_idx]
        plot_y = y[start_idx:end_idx]

    # Plot waveform
    plt.plot(plot_time, plot_y)
    plt.ylabel('Amplitude')
    plt.xlabel('Time (sec)')
    plt.title("Waveform")

    # Set x-axis limits if duration is specified
    if duration is not None:
        plt.xlim(start_time, start_time + duration)

    # Check if beats are provided
    if beats is not None:
        # Filter beats to the time window if duration is specified
        if duration is not None:
            end_time = start_time + duration
            mask = (beats >= start_time) & (beats <= end_time)
            windowed_beats = beats[mask]
            windowed_labels = labels[mask] if labels is not None else None
        else:
            windowed_beats = beats
            windowed_labels = labels

        # Separate beats into downbeats and other beats
        if windowed_labels is not None:
            windowed_labels = np.array(windowed_labels)  # Ensure labels is a NumPy array
            downbeat_times = windowed_beats[windowed_labels == 1]  # Downbeats (label == 1)
            beat_times = windowed_beats[windowed_labels != 1]      # Other beats (label != 1)
            plt.title("Waveform with Beats and Downbeats")
        else:
            downbeat_times = np.array([])  # Empty array for downbeats
            beat_times = windowed_beats    # All beats
            plt.title("Waveform with Beats")

        # Plot vertical lines for downbeats and beats
        ylim = np.max(np.abs(plot_y))
        plt.vlines(downbeat_times, ymin=-ylim-0.1, ymax=ylim+0.1, label='DownBeats', color='black', linewidths=1, linestyle='--')
        plt.vlines(beat_times, ymin=-ylim-0.1, ymax=ylim+0.1, label='Beats', color='red', linewidths=1, linestyle=':')

        plt.legend(frameon=True, framealpha=1.0, edgecolor='black', loc='lower left', bbox_to_anchor=(0, 0.05), fontsize='small')

        # Generate click sounds for downbeats and beats
        # Use the windowed audio for click generation
        audio_length = len(plot_y)

        # Adjust beat times relative to the start of the windowed audio
        if duration is not None:
            adjusted_downbeat_times = downbeat_times - start_time
            adjusted_beat_times = beat_times - start_time

            # Filter out negative times and times beyond the audio length
            time_limit = len(plot_y) / sr
            adjusted_downbeat_times = adjusted_downbeat_times[(adjusted_downbeat_times >= 0) & (adjusted_downbeat_times <= time_limit)]
            adjusted_beat_times = adjusted_beat_times[(adjusted_beat_times >= 0) & (adjusted_beat_times <= time_limit)]
        else:
            adjusted_downbeat_times = downbeat_times
            adjusted_beat_times = beat_times

        downbeat_click = librosa.clicks(times=adjusted_downbeat_times, sr=sr, click_freq=1000, length=audio_length, click_duration=0.1)
        beat_click = librosa.clicks(times=adjusted_beat_times, sr=sr, click_freq=500, length=audio_length, click_duration=0.1)

        # Combine original audio with clicks
        combined_audio = plot_y + beat_click + downbeat_click

        # Normalize combined audio to prevent clipping
        combined_audio = combined_audio / np.max(np.abs(combined_audio))

    else:
        combined_audio = plot_y

    # Play the audio with clicks
    audio_widget = ipd.Audio(combined_audio, rate=sr)

    plt.savefig('plot.png')
    plt.close()

    return audio_widget


def plot_spec(y, sr, gt_beats=None, gt_labels=None, pred_beats=None, pred_labels=None, start_time=0, duration=30):
    """
    Plots spectrogram with optional ground truth and predicted beats/downbeats.

    Parameters:
    - y: Audio signal
    - sr: Sampling rate
    - gt_beats: Ground truth beat times (optional)
    - gt_labels: Ground truth beat labels (optional, 1 for downbeats, other values for beats)
    - pred_beats: Predicted beat times (optional)
    - pred_labels: Predicted beat labels (optional, 1 for downbeats, other values for beats)
    - start_time: Start time in seconds for the plot window (default: 0)
    - duration: Duration in seconds for the plot window (default: 20)
    """
    hop_length = 512

    # Calculate end time
    end_time = start_time + duration

    # Extract audio segment
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    audio_segment = y[start_sample:end_sample]

    # Generate spectrogram
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio_segment, hop_length=hop_length)), ref=np.max)
    librosa.display.specshow(spec, y_axis='log', sr=sr, hop_length=hop_length, x_axis='time')
    plt.title(f"Log-frequency power spectrogram of track ({start_time}s - {end_time}s)")
    plt.colorbar(format="%+2.f dB")

    # Process ground truth beats if provided
    if gt_beats is not None:
        # Filter beat times and labels to the specified window
        gt_mask = (gt_beats >= start_time) & (gt_beats <= end_time)
        gt_beats_windowed = gt_beats[gt_mask]
        gt_labels_windowed = gt_labels[gt_mask] if gt_labels is not None else None

        # Adjust times relative to the start of the window
        gt_beats_adjusted = gt_beats_windowed - start_time

        # Separate beats and downbeats for ground truth
        if gt_labels_windowed is not None:
            gt_labels_windowed = np.array(gt_labels_windowed)
            gt_downbeats = gt_beats_adjusted[gt_labels_windowed == 1]
            gt_beats_only = gt_beats_adjusted[gt_labels_windowed != 1]
        else:
            gt_downbeats = np.array([])
            gt_beats_only = gt_beats_adjusted

        # Plot ground truth annotations in the upper part
        if len(gt_beats_only) > 0:
            plt.vlines(gt_beats_only, hop_length * 2, sr / 2, linestyles='dotted', color='w', alpha=0.8)
        if len(gt_downbeats) > 0:
            plt.vlines(gt_downbeats, hop_length * 2, sr / 2, color='w', alpha=0.8)
        plt.text(duration * 0.35, hop_length * 1.65, 'GT Annotations (above)', color='w', fontsize=10, fontweight='bold')

    # Process predicted beats if provided
    if pred_beats is not None:
        # Filter beat times and labels to the specified window
        pred_mask = (pred_beats >= start_time) & (pred_beats <= end_time)
        pred_beats_windowed = pred_beats[pred_mask]
        pred_labels_windowed = pred_labels[pred_mask] if pred_labels is not None else None

        # Adjust times relative to the start of the window
        pred_beats_adjusted = pred_beats_windowed - start_time

        # Separate beats and downbeats for predictions
        if pred_labels_windowed is not None:
            pred_labels_windowed = np.array(pred_labels_windowed)
            pred_downbeats = pred_beats_adjusted[pred_labels_windowed == 1]
            pred_beats_only = pred_beats_adjusted[pred_labels_windowed != 1]
        else:
            pred_downbeats = np.array([])
            pred_beats_only = pred_beats_adjusted

        # Plot predictions in the lower part
        if len(pred_beats_only) > 0:
            plt.vlines(pred_beats_only, 0, hop_length, linestyles='dotted', color='w', alpha=0.8)
        if len(pred_downbeats) > 0:
            plt.vlines(pred_downbeats, 0, hop_length, color='w', alpha=0.8)
        plt.text(duration * 0.35, hop_length * 1.1, 'Predictions (below)', color='w', fontsize=10, fontweight='bold')

    plt.show()

audio_path = "../raksha_bettare.wav"

print(compiam.list_models())

tracker = compiam.load_model('rhythm:tcn-carnatic')
beats, beat_positions = tracker.predict(audio_path)

print(beats)

y, sr = librosa.load(audio_path)

plot_sonify(y=y, sr=sr, beats=beats)
