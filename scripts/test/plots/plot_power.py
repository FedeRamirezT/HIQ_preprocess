"""
Script to plot clean EEGs

Federico Ramírez-Toraño
13/07/2026

"""

# Imports
import numpy
import matplotlib.pyplot as plt

from init.init import init
from sEEGnal.tools.bids_tools import build_BIDS_object, read_relative_power_spectrum
from scripts.shared.check_valid import check_valid

# Init the database
config, files, sub, ses, task = init()
config['subsystem'] = 'feature_extraction'

# Keep only the valid ones
files, sub, ses, task = check_valid(files, sub, ses, task)

# List of subjects with errors
errors = []

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

    # Load power
    relative_power_spectrum, freqs, metadata = read_relative_power_spectrum(config, BIDS,space='sensor')
    relative_power_spectrum = numpy.transpose(relative_power_spectrum)

    # Plot
    plt.plot(freqs, relative_power_spectrum)
    plt.show(block=True)