# D) Post-Processing Data Analysis Script

EEG Plotting scripts from trials of Mapping Cognitive Control 
Use CSV files generated that contain microvolt information from each lobe under each testing senario 
These scripts were developed using 3 CSV files from lobe stimuli trials (can be developed for 4)


D (a) - Stimulus Trial EEG Comparison
Plots raw EEG data collected across the lobes recorded with stimuli event markers.
Reads per-lobe CSV logs, plots microvolt amplitudes for each lobe on one plot.

D (b) - Stimulus Trial EEG Features Comparison
Plots EEG band features (Theta /Alpha /Beta) across the lobes recorded with 150 s cap 
Reads per-lobe CSV logs, computes per-second mean FFT band amplitudes, and plots three features on each subplot.
