# EEG Plots from trials of Mapping Cognitive Control 
# CSV Files containing microvolt information from each lobe under each testing senario 
#__________________________________________________________________________________________
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# File paths
file_paths = [
    "eeg_stream_log_08_08_TEMPROB.csv",
    "eeg_stream_log_08_08_PARIETALROB.csv",
    "eeg_stream_log_08_08_FRONTALROB.csv",
]
labels = ["Temporal", "Parietal", "Frontal"]

def process_file(path):
    df = pd.read_csv(path)
    # Convert Timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S')
    # Calculate relative time in seconds
    df['Time_s'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds()
    # Cap to 150 seconds
    df = df[df['Time_s'] <= 150]
    # Compute mean amplitude across EEG channels
    df['Mean_uV'] = df.drop(columns=['Timestamp', 'Time_s']).mean(axis=1)
    return df['Time_s'], df['Mean_uV']

# Process each file
plot_data = [process_file(file_paths[i]) for i in range(len(file_paths))]

# Build comparison DataFrame
comparison_df = pd.DataFrame({
    "Time_s": plot_data[0][0],  # Use the first file's time as reference
    labels[0]: plot_data[0][1],
    labels[1]: plot_data[1][1],
    labels[2]: plot_data[2][1]
})

# Save trimmed version as CSV
comparison_df.to_csv("eeg_stream_log_comparison.csv", index=False)
print("Capped comparison CSV saved as raw_eeg_stream_log_comparison.csv")

# ________________________________________________________

# Plot EEG with vertical stimulus markers
plt.figure(figsize=(10, 6))
for i, label in enumerate(labels):
    plt.plot(plot_data[i][0], plot_data[i][1], label=label)

# Add event markers
markers = [30, 90, 120]
for mark in markers:
    plt.axvline(x=mark, color='blue', linestyle='--', linewidth=1)
    plt.text(mark + 1, plt.ylim()[1]*0.95, f"{mark}s", color='blue')

# Styling
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.title("EEG Stream Comparison with Event Markers")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save and show
plt.savefig("eeg_stream_comparison_plot.png", dpi=300)
plt.show()
