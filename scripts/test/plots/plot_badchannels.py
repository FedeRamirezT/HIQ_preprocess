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
        recordings = [recordings[59]]
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
                'components_to_include': [],
                'components_to_exclude': ['eog', 'ecg']
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

            channels_metadata = bids_tools.read_channels(config, BIDS)
            metadata_bad_names = set(
                channels_metadata.loc[
                    channels_metadata['status'] == 'bad',
                    'name',
                ]
            )

            # Preserve Raw channel order; set intersection made this metadata
            # nondeterministic between processes.
            badchannels = [
                channel
                for channel in raw.ch_names
                if channel in metadata_bad_names
            ]

            raw = mne_tools.prepare_eeg(
                config,
                BIDS,
                raw=raw,
                channels_to_include=badchannels
            )

            # Get correlation
            data_unfiltered = raw.get_data()
            corr_unfiltered = numpy.corrcoef(data_unfiltered)
            mask_unfiltered = corr_unfiltered > config['preprocess']['badchannel_detection']['gel_bridge']['threshold']

            raw = mne_tools.prepare_eeg(
                config,
                BIDS,
                raw=raw,
                freq_limits=[2,45]
            )

            # Get correlation
            data_filtered = raw.get_data()
            corr_filtered = numpy.corrcoef(data_filtered)
            mask_filtered = corr_filtered > config['preprocess']['badchannel_detection']['gel_bridge']['threshold']

            raw.plot(block=True)






if __name__ == '__main__':
    main()
