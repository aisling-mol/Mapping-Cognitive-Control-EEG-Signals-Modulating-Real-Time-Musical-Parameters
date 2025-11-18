# VSVersionofCode_LobeSpecific_V3.3.py
# Real-Time EEG Band Power to MIDI CC Conversion with Filters & Serial Streaming
#________________________________________________________________________________

# Adjust date of CSV file name ; Line 77

# Import Libraries
#````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
import numpy as np
import mido
import time
import serial
import csv
from scipy.signal import butter, filtfilt

# Serial Port Configuration
#````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
serial_port = 'COM8'
baud_rate = 115200
sfreq = 250
chunk_samples = sfreq
num_channels = 8
midi_port_name = 'loopMIDI Port 1'


# Mapping & Scaling
#````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

# Channel to lobe mapping
channel_lobe_map = {
    "Occipital": [0, 1],
    "Temporal": [2, 3],
    "Central": [4, 5],
    "Frontal": [6, 7]
}

# Active lobes toggle
active_lobes = {
    "Frontal": True,
    "Central": False,
    "Temporal": False,
    "Occipital": False
}

# MIDI CC numbers per lobe-band
cc_mapping = {
    "Frontal":   {"Theta": 16, "Alpha": 17, "Beta": 18},
    "Central":   {"Theta": 19, "Alpha": 20, "Beta": 21},
    "Temporal":  {"Theta": 22, "Alpha": 23, "Beta": 24},
    "Occipital": {"Theta": 25, "Alpha": 26, "Beta": 27}
}

# EEG bands
bands = {
    "Theta": (4.0, 7.0),
    "Alpha": (8.0, 12.0),
    "Beta": (13.0, 30.0)
}

scale_factor = 0.0025


# Filtering & Setups
#````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

# Filter design
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs  # Nyquist limit set
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

def apply_bandpass(data, lowcut=1.6, highcut=40.0, fs=250):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data)

# Serial port setup
try:
    ser = serial.Serial(serial_port, baud_rate, timeout=1)
    print(f"Listening on {serial_port} at {baud_rate} baud...\n")
except Exception as e:
    raise SystemExit(f"Could not open serial port: {e}")

# MIDI setup
outport = mido.open_output(midi_port_name)
print(f"MIDI output port '{midi_port_name}' opened.\n")

# CSV logging setup
csv_file = open("eeg_stream_log_rename_here.csv", "w", newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp"] + [f"Ch{i}" for i in range(num_channels)])

# Parse function
def parse_serial_line(line):
    try:
        # Decode and strip control characters
        line = line.decode(errors='ignore').strip()

        # Check format: starts with $, ends with ;
        if not (line.startswith('$') and line.endswith(';')):
            return None

        # Remove '$' and ';', then split by space
        body = line[1:-1].strip()
        parts = body.split()

        # Convert all to integers, allow negative numbers and zeros
        if len(parts) != num_channels:
            return None

        return [int(p) for p in parts]

    except Exception as e:
        print(f"Parse error: {e}")
        return None


# Main Loop
#````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

print("Streaming EEG data...\n")
while True:
    eeg_buffer = [[] for _ in range(num_channels)]
    start_time = time.time()

    while all(len(ch_data) < chunk_samples for ch_data in eeg_buffer):
        try:
            line = ser.read_until(b';')
            values = parse_serial_line(line)

            # Ensure valid and complete sample
            if values is not None:
                for ch in range(num_channels):
                    eeg_buffer[ch].append(values[ch])
                timestamp = time.strftime("%H:%M:%S", time.localtime())
                csv_writer.writerow([timestamp] + values)
            else:
                print(f"Incomplete or invalid line skipped: {line}")

        except Exception as e:
            print(f"Serial read error: {e}")

    for i, ch in enumerate(eeg_buffer):
        if len(ch) != chunk_samples:
            print(f"Channel {i} only has {len(ch)} samples!")
    
    eeg_data = np.array(eeg_buffer)  # shape (8, 250)

    for lobe, ch_indices in channel_lobe_map.items():
        if not active_lobes.get(lobe, False):
            continue

        valid_channels = [eeg_data[i] for i in ch_indices if not np.all(eeg_data[i] == 0)]
        if not valid_channels:
            print(f"{lobe} skipped: all channels zero.")
            continue

        lobe_signal = np.mean(valid_channels, axis=0)
        raw_mean = int(np.mean(lobe_signal))
        print(f"{lobe} μV mean: {raw_mean}")

        filtered = apply_bandpass(lobe_signal, 1.6, 40.0, sfreq)
        filtered -= np.mean(filtered)

        freqs = np.fft.rfftfreq(chunk_samples, d=1.0 / sfreq)
        fft_vals = np.fft.rfft(filtered)
        magnitudes = np.abs(fft_vals)

        for band_name, (f_low, f_high) in bands.items():
            idx = (freqs >= f_low) & (freqs <= f_high)
            amp = magnitudes[idx].mean() if np.any(idx) else 0.0
            midi_val = int(np.clip(amp * scale_factor, 0, 127))
            cc_num = cc_mapping[lobe][band_name]
            outport.send(mido.Message('control_change', control=cc_num, value=midi_val))
            print(f"Sent MIDI CC{cc_num} ({lobe}-{band_name}): {midi_val}")

    elapsed = time.time() - start_time
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)
