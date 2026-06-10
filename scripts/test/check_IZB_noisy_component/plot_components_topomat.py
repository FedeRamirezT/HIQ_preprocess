"""
Script to call automatic preprocess for all EEGs

Federico Ramírez-Toraño
28/10/2025

"""

# Imports
import os
from init.init import init
from sEEGnal.tools.bids_tools import build_BIDS_object
import sEEGnal.tools.bids_tools as bids_tools
import sEEGnal.tools.mne_tools as mne_tools
import matplotlib.pyplot as plt

# Fix plot
import matplotlib
matplotlib.use('Qt5Agg')

# Init the database
config, files, sub, ses, task = init()
config['subsystem'] = 'preprocess'

# Output folder for figures
config['path']['IC_topoplot'] = os.path.join('scripts', 'test', 'check_IZB_noisy_component','components_figures','IC_topoplot')

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

    # Read SOBI
    # Load the clean EEG
    apply_sobi = {
        'desc': 'sobi',
        'components_to_include': ['brain', 'other'],
        'components_to_exclude': []
    }

    freq_limits = [
        config['component_estimation']['low_freq'],
        config['component_estimation']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    epoch_definition = config['source_reconstruction']['epoch_definition']

    # Load the clean data
    raw = mne_tools.prepare_eeg(
        config,
        BIDS,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        freq_limits=freq_limits,
        notch_filter=True,
        resample_frequency=resample_frequency,
        set_annotations=True,
        crop_seconds=crop_seconds,
        rereference='average'
    )
    sobi = bids_tools.read_sobi(config, BIDS, raw, apply_sobi['desc'])

    # Save the IC figures
    for i in range(sobi.mixing_matrix_.shape[1]):

        # Create the output file
        output_file = os.path.join(
            config['path']['IC_topoplot'],
            current_sub,
            f"{current_sub}_IC_{i}.png"
        )

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Plot and save
        figs = sobi.plot_properties(
            raw,
            picks=i,
            dB=False,
            show=False
        )

        figs[0].savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(figs[0])
