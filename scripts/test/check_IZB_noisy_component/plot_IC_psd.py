# Imports
import os

import h5py
import numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from init.init import init
import sEEGnal.tools.bids_tools as bids_tools
import sEEGnal.tools.mne_tools as mne_tools

# Init the database
config, files, sub, ses, task = init()

# Go through each subject
index = range(len(files))
index = [8]
for current_index in index:

    # current info
    current_file = files[current_index]
    current_sub = sub[current_index]
    current_ses = ses[current_index]
    current_task = task[current_index]

    # Create the subjects following AI-Mind protocol
    BIDS = bids_tools.build_BIDS_object(config, current_sub, current_ses, current_task)

    print('Working with sub ' + current_sub + ' ses ' + current_ses + ' task ' + current_task)

    # Load low and high noisy ICs
    fname = os.path.join('scripts', 'shared', 'templates', 'template_low_freq_noisy_IC.h5')
    with h5py.File(fname, 'r') as h5:
        freqs_low = h5['freqs'][:]
        templates_low = h5['templates'][:]
        metadata_low = h5['metadata'][:]

    # Parameters for loading EEG recordings
    config['subsystem'] = 'preprocess'
    freq_limits = [
        config['component_estimation']['low_freq'],
        config['component_estimation']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    epoch_definition = {
        'mode': 'fixed',
        'length': 4,
        'overlap': 0,
        'padding': 0,
        'reject_by_annotation': 1
    }
    set_annotations = True

    # Load raw EEG
    raw = mne_tools.prepare_eeg(
        config,
        BIDS,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        notch_filter=True,
        freq_limits=freq_limits,
        resample_frequency=resample_frequency,
        metadata_badchannels=True,
        interpolate_badchannels=True,
        set_annotations=set_annotations,
        crop_seconds=crop_seconds,
        rereference='average',
        epoch_definition=epoch_definition
    )

    # Read SOBI
    sobi = bids_tools.read_sobi(config, BIDS, raw, 'sobi')
    ICs_time_series = sobi.get_sources(raw)

    # Outputs
    r_all_low = []
    index_low = []
    psd_low = []
    for iIC in range(len(ICs_time_series.picks)):

        # Get the current channel
        current_IC_time_series = ICs_time_series.copy().pick(iIC)

        ###################
        # Low freq IC
        ###################

        # Estimate spectrum
        n_samples = current_IC_time_series.get_data().shape[-1]
        n_fft = min(2048, n_samples)
        n_per_seg = n_fft
        n_overlap = n_fft // 2
        spectrum = current_IC_time_series.compute_psd(
            method='welch',
            fmin=2,
            fmax=45,
            picks='all',
            n_fft=n_fft,
            n_per_seg=n_per_seg,
            n_overlap=n_overlap
        ).average()
        psd = spectrum.get_data(picks='all')[0]
        psd = numpy.log10(psd + numpy.finfo(float).eps)
        psd = (psd - psd.mean()) / psd.std()

        # Compare to the low ICs
        for current_template in templates_low:

            r = numpy.corrcoef(current_template, psd)
            r_all_low.append(r[0, 1])

            # Save the important ones
            if r[0, 1] > 0.75:
                index_low.append(iIC)
                psd_low.append(psd)
            else:
                plt.plot(freqs_low, psd, color='grey')

    for current in psd_low:
        plt.plot(freqs_low, current,color='r')

    plt.show(block=True)
