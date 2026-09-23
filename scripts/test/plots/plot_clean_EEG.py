"""
Script to plot clean EEGs

Federico Ramírez-Toraño
13/07/2026

"""


"""
Plot the EEG recording with the bad channels marked for visual inspection
"""

from pathlib import Path
import sys


if __package__ in (None, ''):
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root))

import numpy

import scripts.test.plots.init.init as init
import sEEGnal
import sEEGnal.tools.mne_tools as mne_tools
import sEEGnal.tools.bids_tools as bids_tools



def main():

        # Load configuration and recordings
        config = init.load_config()
        recordings =  sEEGnal.io.recordings .validate_recordings(config)
        recordings = [recordings[2]]
        for recording in recordings:
            print(
                f"Working with sub {recording['subject']} "
                f"ses {recording['session']} task {recording['task']} "
                f"run {recording['run']}"
            )

            BIDS = recording['bids_path']

            # Parameters for loading EEG  recordings
            config['subsystem'] = 'preprocess'
            ica_config = {
                'desc': 'badchannels',
                'components_to_include': ['brain','other'],
                'components_to_exclude': []
            }
            resample_frequency = config['component_estimation']['resample_frequency']
            channels_to_include = config['global']['channels_to_include']
            channels_to_exclude = config['global']['channels_to_exclude']
            crop_seconds = config['preprocess']['badchannel_detection']['crop_seconds']

            # Load the raw EEG
            raw = mne_tools.prepare_eeg(config, BIDS, preload=True, channels_to_include=channels_to_include, channels_to_exclude=channels_to_exclude, resample_frequency=resample_frequency, notch_filter=True, crop_seconds=crop_seconds)

            # Apply IC
            raw = mne_tools.apply_ica(
                config,
                BIDS,
                raw,
                ica_config
            )

            # Add badchannels and filter
            raw = mne_tools.prepare_eeg(
                config,
                BIDS,
                raw=raw,
                freq_limits=[2, 45],
                metadata_badchannels=True,
                set_annotations=True
            )

            raw.plot(block=True)




if __name__ == '__main__':
    main()








'''
# Imports
from init.init import init
from sEEGnal.tools.bids_tools import build_BIDS_object
from sEEGnal.tools.mne_tools import prepare_eeg

# Init the database
config, files, sub, ses, task = init()

# List of subjects with errors
errors = []

# Go through each subject
index = range(len(files))
index = [50]
for current_index in index:

    # current info
    current_file = files[current_index]
    current_sub = sub[current_index]
    current_ses = ses[current_index]
    current_task = task[current_index]

    # Create the subjects following AI-Mind protocol
    BIDS = build_BIDS_object(config, current_sub, current_ses, current_task)

    print('Working with sub ' + current_sub + ' ses ' + current_ses + ' task ' + current_task)

    # Load the clean EEG
    sobi = {
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

    # Load the clean data
    config['subsystem'] = 'preprocess'
    raw = prepare_eeg(
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

    raw = prepare_eeg(
        config,
        BIDS,
        raw=raw,
        apply_sobi=sobi,
        freq_limits=[2, 45],
        metadata_badchannels=True
    )

    raw.plot(block=True)
'''