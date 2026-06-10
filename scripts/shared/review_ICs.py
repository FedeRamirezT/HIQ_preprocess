"""
There are two noisy components in this dataset due to the experiment room.
Review the ICs to look for these components.

Federico Ramírez-Toraño
10/06/2026

"""

# Imports
import os

import h5py
import numpy
import matplotlib.pyplot as plt

import sEEGnal.tools.mne_tools as mne_tools
import sEEGnal.tools.bids_tools as bids_tools
from sEEGnal.tools.qc_tools import artifact_qc


def review_ICs(config, BIDS):

    # Load low and high noisy ICs
    fname = os.path.join('scripts','shared','templates','template_low_freq_noisy_IC.h5')
    with h5py.File(fname, 'r') as h5:
        freqs_low = h5['freqs'][:]
        templates_low = h5['templates'][:]
        metadata_low = h5['metadata'][:]

    fname = os.path.join('scripts', 'shared', 'templates', 'template_high_freq_noisy_IC.h5')
    with h5py.File(fname, 'r') as h5:
        freqs_high = h5['freqs'][:]
        templates_high = h5['templates'][:]
        metadata_high = h5['metadata'][:]


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
    r_all_high = []
    index_high = []
    psd_low = []
    psd_high = []
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

            r = numpy.corrcoef(current_template,psd)
            r_all_low.append(r[0,1])

            # Save the important ones
            if r[0,1] > 0.8:
                index_low.append(iIC)
                psd_low.append(psd)

        ###################
        # High freq IC
        ###################
        '''
        spectrum = current_IC_time_series.compute_psd(
            method='welch',
            fmin=60,
            fmax=100,
            picks='all',
            n_fft=n_fft,
            n_per_seg=n_per_seg,
            n_overlap=n_overlap
        ).average()
        psd = spectrum.get_data(picks='all')[0]
        psd = numpy.log10(psd + numpy.finfo(float).eps)
        psd = (psd - psd.mean()) / psd.std()

        # Compare to the high ICs
        for current_template in templates_high:
            r = numpy.corrcoef(current_template, psd)
            r_all_high.append(r[0, 1])

            # Save the important ones
            if r[0, 1] > 0.8:
                index_high.append(iIC)
                psd_high.append(psd)
        '''

    # Change the original labels if needed
    labels = ['brain', 'muscle', 'eog', 'ecg', 'line_noise', 'ch_noise']

    # Low
    if len(index_low) > 0:

        # Remove replicates
        index_low = list(set(index_low))

        for iIC in index_low:

            # Remove from the original category
            for current_label in labels:

                if iIC in sobi.labels_[current_label]:

                    sobi.labels_[current_label].remove(iIC)
                    break

            sobi.labels_['line_noise'].append(iIC)

    # High
    '''
    if len(index_high) > 0:

        # Remove replicates
        index_high = list(set(index_high))

        for iIC in index_high:

            # Remove from the original category
            for current_label in labels:

                if iIC in sobi.labels_[current_label]:
                    sobi.labels_[current_label].remove(iIC)
                    break

            sobi.labels_['line_noise'].append(iIC)
    '''


    # Save the new SOBI
    _ = bids_tools.write_sobi(config, BIDS, sobi, 'sobi')

    # Redo the QC figures
    artifact_qc(config, BIDS)



    '''
    # Convert to array
    r_all_low = numpy.asarray(r_all_low)
    psd_low = numpy.asarray(psd_low).T
    freqs_low = numpy.asarray(freqs_low)
    r_all_high = numpy.asarray(r_all_high)
    psd_high = numpy.asarray(psd_high).T
    freqs_high = numpy.asarray(freqs_high)

    plt.figure()
    # Plot low
    plt.subplot(2, 2, 1)
    plt.plot(numpy.abs(r_all_low))
    plt.subplot(2, 2, 2)
    plt.plot(freqs_low, psd_low)
    # Plot high
    plt.subplot(2, 2, 3)
    plt.plot(numpy.abs(r_all_high))
    plt.subplot(2, 2, 4)
    plt.plot(freqs_high, psd_high)
    plt.show(block=True)
    '''