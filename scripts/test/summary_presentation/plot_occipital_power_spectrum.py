
# Imports
import os

import numpy
import matplotlib.pyplot as plt

from init.init import init
from sEEGnal.tools.bids_tools import build_BIDS_object, read_relative_power_spectrum
from scripts.shared.check_valid import check_valid

# Fix qtagg
import matplotlib
matplotlib.use('tkagg')


# Init the database
config, files, sub, ses, task = init()
config['subsystem'] = 'feature_extraction'

# Keep only the valid ones
files, sub, ses, task = check_valid(files, sub, ses, task)

# Occipital sensors
occipital_ch_names = [ 'O1', 'O2', 'PO3', 'PO4', 'PO7', 'PO8', 'POz']

# Create the figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

# Go through each subject
index = range(len(files))
for current_index in index:

    # current info
    current_file = files[current_index]
    current_sub = sub[current_index]
    current_ses = ses[current_index]
    current_task = task[current_index]

    # Create the subjects following AI-Mind protocol
    BIDS = build_BIDS_object(config, current_sub, current_ses, current_task)

    print('Working with sub ' + current_sub + ' ses ' + current_ses + ' task ' + current_task)

    # Load the power spectrum data and metadata
    relative_power_spectrum, freqs, metadata = read_relative_power_spectrum(config,BIDS,'sensor')

    # Select occipital channels
    occipital_index = [i for i in range(len(metadata["ch_names"])) if metadata['ch_names'][i] in occipital_ch_names]

    # Average pow spectrum
    relative_power_spectrum_mean = numpy.mean(relative_power_spectrum[occipital_index,:],axis=0)

    # Get 2-45 Hz
    freqs_mask = (freqs > 2) & (freqs < 45)
    freqs = freqs[freqs_mask]
    relative_power_spectrum_mean = relative_power_spectrum_mean[freqs_mask]

    # Plot
    ax.plot(freqs, relative_power_spectrum_mean)

# Plot details
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Relative power")
plt.tight_layout()

figure_path = os.path.join('scripts','test','summary_presentation','figures','all_occipital_power_spectrum.png')
plt.savefig(figure_path)
plt.close()