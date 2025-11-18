# EEG per-lobe band features plotter (Theta/Alpha/Beta) with 150 s cap ===
# Reads per-lobe CSV logs, computes per-second band amplitudes, and plots 3 subplots.

# Import Libraries
#`````````````````````````````````````````````````````````````````````````````````````
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


# CSV INPUTS
# ------------------------------------------------------------
file_paths = [
    "eeg_stream_log_08_08_TEMPROB.csv",
    "eeg_stream_log_08_08_PARIETALROB.csv",
    "eeg_stream_log_08_08_FRONTALROB.csv",
]
labels = ["Temporal", "Parietal", "Frontal"]

fs = 250                   # Hz
max_time_s = 150           # Cap analysis to 150 seconds
samp_per_sec = fs

channel_lobe_map = {
    "Occipital": [0, 1],
    "Temporal":  [2, 3],
    "Parietal":  [4, 5],
    "Frontal":   [6, 7],
}

bands = {
    "Theta": (4.0, 7.0),
    "Alpha": (8.0, 12.0),
    "Beta":  (13.0, 30.0)
}

bp_low, bp_high = 1.6, 40.0
event_marks_s = [30, 90, 120]

# DSP
# ------------------------------------------------------------
def butter_bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return b, a

def apply_bandpass(x, low=bp_low, high=bp_high, fs=fs):
    if len(x) < 5:
        return x
    b, a = butter_bandpass(low, high, fs)
    return filtfilt(b, a, x)

def compute_band_amplitudes(signal_1d, fs=fs, bands=bands):
    seg = signal_1d - np.mean(signal_1d)
    freqs = np.fft.rfftfreq(len(seg), d=1.0/fs)
    mags  = np.abs(np.fft.rfft(seg))
    return {name: float(mags[(freqs >= f_lo) & (freqs <= f_hi)].mean()) if np.any((freqs >= f_lo) & (freqs <= f_hi)) else 0.0
            for name, (f_lo, f_hi) in bands.items()}

def load_and_features(csv_path, lobe_name, lobe_channel_map, fs=fs):
    df = pd.read_csv(csv_path)
    ch_cols = [c for c in df.columns if c.startswith("Ch")]
    if len(ch_cols) < 8:
        raise ValueError(f"{csv_path}: expected 8 channel columns named Ch0..Ch7.")

    ch_idx = lobe_channel_map.get(lobe_name, [])
    if not ch_idx:
        raise ValueError(f"No channel mapping for lobe '{lobe_name}'.")

    for i in ch_idx:
        if f"Ch{i}" not in df.columns:
            raise ValueError(f"{csv_path}: missing Ch{i} for {lobe_name}.")

    # Limit to first MAX_TIME_S seconds of data
    max_samples = max_time_s * fs
    if len(df) > max_samples:
        df = df.iloc[:max_samples]

    lobe_matrix = df[[f"Ch{i}" for i in ch_idx]].to_numpy(dtype=float)
    lobe_signal = np.mean(lobe_matrix, axis=1)
    lobe_filt = apply_bandpass(lobe_signal, bp_low, bp_high, fs)

    n_win = len(lobe_filt) // fs
    times = np.arange(n_win, dtype=float)
    feats = {name: [] for name in bands.keys()}

    for w in range(n_win):
        seg = lobe_filt[w*fs:(w+1)*fs]
        band_vals = compute_band_amplitudes(seg, fs, bands)
        for name in bands.keys():
            feats[name].append(band_vals[name])

    return times, feats

# Run & plot
# ------------------------------------------------------------
results = []
for path, lobe in zip(file_paths, labels):
    t_s, band_feats = load_and_features(path, lobe, channel_lobe_map, fs)
    results.append((lobe, t_s, band_feats))

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)

for ax, (lobe, t_s, feats) in zip(axes, results):
    ax.plot(t_s, feats["Theta"], label="Theta (4–7 Hz)")
    ax.plot(t_s, feats["Alpha"], label="Alpha (8–12 Hz)")
    ax.plot(t_s, feats["Beta"],  label="Beta (13–30 Hz)")
    ax.set_title(f"{lobe} Lobe Band Amplitudes over Time")
    ax.set_ylabel("Mean FFT Magnitudes (µV)")
    ax.grid(True, alpha=0.3)
    for m in event_marks_s:
        ax.axvline(x=m, linestyle="--", linewidth=1, color="red")
        ax.text(m + 0.5, ax.get_ylim()[1] * 0.92, f"{m}s", color="red")

axes[-1].set_xlabel("Time (s)")
axes[0].legend(loc="upper right")
fig.tight_layout()
plt.savefig("per_lobe_band_features_capped.png", dpi=300)
plt.show()
