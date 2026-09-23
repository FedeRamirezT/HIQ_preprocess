"""
Script to save segments of clean EEG data
Federico Ramírez-Toraño
13/07/2026

"""


from pathlib import Path
import sys
import os


if __package__ in (None, ''):
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root))

import scripts.test.plots.init.init as init
import sEEGnal
import sEEGnal.tools.mne_tools as mne_tools
import sEEGnal.tools.bids_tools as bids_tools



def main():

        # Load configuration and recordings
        config = init.load_config()
        recordings =  sEEGnal.io.recordings .validate_recordings(config)
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

            epoch_definition = {
                "mode": "fixed",
                "duration": 4,
                "overlap": 0,
                "reject_by_annotation": True
            }

            # Add badchannels, annotations and build epochs
            epochs = mne_tools.prepare_eeg(
                config,
                BIDS,
                raw=raw,
                preload=True,
                metadata_badchannels=True,
                interpolate_badchannels=True,
                set_annotations=True,
                epoch_definition=epoch_definition
            )


            # Creates the derivatives folder, if required.
            outfile = bids_tools.build_derivatives_path(BIDS,'export','clean_segments.set')
            derivatives_folder_path = os.path.dirname(outfile)
            if not os.path.isdir(derivatives_folder_path):
                os.makedirs(derivatives_folder_path)

            # Export the clean Epoch object
            epochs.export(
                outfile,
                fmt="eeglab",
                overwrite=True
            )

            print()





if __name__ == '__main__':
    main()
