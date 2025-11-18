# VSVersionofCode_V1.0.py
# Real-Time EEG Band Power to MIDI CC Conversion Script V1.0
#____________________________________________________________________________________________________________________

# Import Libraries
#````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

import mne
import numpy as np
import mido
import time


# Load Data
#````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

# Path to your EEG EDF file
edf_path = 'C:/Users/Windows 10/OneDrive/Documents/OpenNeuro/sub-1/eeg/sub-1_task-oa_eeg.edf'

# Load the raw EDF data
raw = mne.io.read_raw_edf(edf_path, preload=True)
print(f"Loaded data with {raw.info['nchan']} channels and duration {raw.times[-1]:.2f} seconds.")

# Select only EEG channels (drop others like EOG or annotations)
raw_eeg = raw.copy().pick_types(eeg=True)
print(f"EEG channels selected: {raw_eeg.ch_names}")

# Define channel groups by lobe based on name prefixes
channel_groups = {
    "Frontal":   [ch for ch in raw_eeg.ch_names if ch.upper().startswith("EEG F") or ch.upper().startswith("F")],
    "Central":   [ch for ch in raw_eeg.ch_names if ch.upper().startswith("EEG C") or ch.upper().startswith("C")],
    "Temporal":  [ch for ch in raw_eeg.ch_names if ch.upper().startswith("EEG T") or ch.upper().startswith("T")],
    "Parietal":  [ch for ch in raw_eeg.ch_names if ch.upper().startswith("EEG P") or ch.upper().startswith("P")],
    "Occipital": [ch for ch in raw_eeg.ch_names if ch.upper().startswith("EEG O") or ch.upper().startswith("O")]
}

print("Channel groups by lobe:", {lobe: len(chs) for lobe, chs in channel_groups.items()})

# For future use: could iterate over channel_groups to process each lobe separately.
# For now, combining all EEG channels for a single analysis.
data = raw_eeg.get_data()               # shape is (n_channels, n_samples)
combined_signal = data.mean(axis=0)     # average across all EEG channels

# Get Info About Dataset
#````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
 
print(f"Sampling Frequency of Set: {raw.info['sfreq']}")

print(f"EEG Sample Data: {raw_eeg.get_data(0, 120)}")

# Set up MIDI Output - Virtual port
#````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
 
midi_port_name = 'loopMIDI Port 1'

# Open available chosen output port
outport = mido.open_output(midi_port_name)
print(f"MIDI output port '{midi_port_name}' opened. Send this port to Reaper as MIDI input.")

# Define frequency band ranges (Hz)
theta_band = (4.0, 7.0)
alpha_band = (8.0, 12.0)
beta_band  = (13.0, 30.0)

# Assign MIDI CC numbers for each band
cc_theta = 16  # CC21 for Theta
cc_alpha = 17  # CC22 for Alpha
cc_beta  = 18  # CC23 for Beta


# Simulate Live EEG Stream (1-second Chunks)
#````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

sfreq = int(raw.info['sfreq'])  # sampling frequency (Hz)
chunk_samples = sfreq * 1       # number of samples in 1-second chunk
total_samples = combined_signal.shape[0]

print("Starting real-time simulation...")
for start in range(0, total_samples, chunk_samples):
    end = start + chunk_samples
    if end > total_samples:
        break  # ignore last partial chunk if any
    
    chunk = combined_signal[start:end]
    # Remove DC offset (mean) from the chunk
    chunk = chunk - np.mean(chunk)

    # Compute FFT of the 1-second chunk
    freqs = np.fft.rfftfreq(len(chunk), d=1.0/sfreq)
    fft_vals = np.fft.rfft(chunk)
    magnitudes = np.abs(fft_vals)

    # Test FFT Values

    #print(f"FFT Freq Values Sample Data: {freqs}")
    #print(f"FFT Values EEG Sample Data: {fft_vals}")

    # Compute average magnitude in each frequency band
    theta_idx = (freqs >= theta_band[0]) & (freqs <= theta_band[1])
    alpha_idx = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
    beta_idx  = (freqs >= beta_band[0])  & (freqs <= beta_band[1])
    theta_amp = magnitudes[theta_idx].mean()
    alpha_amp = magnitudes[alpha_idx].mean()
    beta_amp  = magnitudes[beta_idx].mean()

    # Scale amplitudes to MIDI 0-127 range

    # Scaling factor for normalisation as needed
    scale_factor = 1e+05
    theta_val = int(np.clip(theta_amp * scale_factor, 0, 127))
    alpha_val = int(np.clip(alpha_amp * scale_factor, 0, 127))
    beta_val  = int(np.clip(beta_amp  * scale_factor, 0, 127))
    # Send MIDI Control Change messages for each band
    outport.send(mido.Message('control_change', control=cc_theta, value=theta_val))
    outport.send(mido.Message('control_change', control=cc_alpha, value=alpha_val))
    outport.send(mido.Message('control_change', control=cc_beta,  value=beta_val))
    # Print the values for debugging and monitoring
    print(f"t={start/sfreq:.1f}s – Theta:{theta_val}, Alpha:{alpha_val}, Beta:{beta_val}")
    # Wait for 1 second before next chunk (simulating live stream timing)
    time.sleep(1)

print("EEG-to-MIDI streaming finished.")
