"""
There are two noisy components in this dataset due to the experiment room.

Create a template for each one

Federico Ramírez-Toraño
10/06/2026
"""

# Imports
from init.init import init
import sEEGnal.tools.bids_tools as bids_tools
import sEEGnal.tools.mne_tools as mne_tools
import numpy
import h5py


# Define the noisy components
low_freq_noisy_IC = {
    '023': [7],
    '024': [8],
    '025': [6],
    '029': [5],
    '031': [5],
    '032': [6],
    '033': [9],
    '034': [5],
    '035': [4],
    '036': [3],
    '037': [6],
    '038': [8],
    '039': [6],
    '040': [8],
    '042': [5],
    '043': [19],
    '044': [17],
    '045': [7],
    '046': [11],
    '047': [34]
}

high_freq_noisy_IC = {
    '023': [52, 53, 54, 55, 56, 57],
    '026': [51, 52, 53, 54, 55, 56, 57],
    '028': [52, 53, 54, 55, 56, 57],
    '029': [49, 50, 51, 52, 53, 54, 55, 56, 57],
    '047': [55, 56, 57]
}


def compute_ic_psd_template(
    config,
    files,
    sub,
    ses,
    task,
    noisy_ic_dict,
    fmin,
    fmax
):
    templates = []
    metadata = []
    freqs_template = None

    for current_index in range(len(files)):

        current_file = files[current_index]
        current_sub = sub[current_index]
        current_ses = ses[current_index]
        current_task = task[current_index]

        if current_sub not in noisy_ic_dict:
            continue

        print(f'Working with sub {current_sub} ses {current_ses} task {current_task}')

        BIDS = bids_tools.build_BIDS_object(
            config,
            current_sub,
            current_ses,
            current_task
        )

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
            set_annotations=True,
            crop_seconds=crop_seconds,
            rereference='average',
            epoch_definition=epoch_definition
        )

        sobi = bids_tools.read_sobi(config, BIDS, raw, 'sobi')
        ICs_time_series = sobi.get_sources(raw)

        for ic in noisy_ic_dict[current_sub]:

            current_IC_time_series = ICs_time_series.copy().pick(ic)

            n_samples = current_IC_time_series.get_data().shape[-1]

            n_fft = min(2048, n_samples)
            n_per_seg = n_fft
            n_overlap = n_fft // 2

            spectrum = current_IC_time_series.compute_psd(
                method='welch',
                fmin=fmin,
                fmax=fmax,
                picks='all',
                n_fft=n_fft,
                n_per_seg=n_per_seg,
                n_overlap=n_overlap
            ).average()

            psd = spectrum.get_data(picks='all')[0]
            freqs = spectrum.freqs

            if freqs_template is None:
                freqs_template = freqs.copy()
            elif not numpy.array_equal(freqs_template, freqs):
                raise ValueError(
                    f'Frequency vector mismatch in sub {current_sub}, IC {ic}'
                )

            psd = numpy.log10(psd + numpy.finfo(float).eps)
            psd = (psd - psd.mean()) / psd.std()

            templates.append(psd)
            metadata.append((current_sub, current_ses, current_task, ic))

    templates = numpy.array(templates)
    metadata = numpy.array(metadata, dtype='S')

    return freqs_template, templates, metadata


def save_template_h5(output_file, freqs, templates, metadata):
    with h5py.File(output_file, 'w') as h5:
        h5.create_dataset('freqs', data=freqs)
        h5.create_dataset('templates', data=templates)
        h5.create_dataset('metadata', data=metadata)


# Init the database
config, files, sub, ses, task = init()
config['subsystem'] = 'preprocess'


# Low-frequency template
low_freqs, low_templates, low_metadata = compute_ic_psd_template(
    config=config,
    files=files,
    sub=sub,
    ses=ses,
    task=task,
    noisy_ic_dict=low_freq_noisy_IC,
    fmin=2,
    fmax=45
)

save_template_h5(
    output_file='template_low_freq_noisy_IC.h5',
    freqs=low_freqs,
    templates=low_templates,
    metadata=low_metadata
)


# High-frequency template
high_freqs, high_templates, high_metadata = compute_ic_psd_template(
    config=config,
    files=files,
    sub=sub,
    ses=ses,
    task=task,
    noisy_ic_dict=high_freq_noisy_IC,
    fmin=60,
    fmax=100
)

save_template_h5(
    output_file='template_high_freq_noisy_IC.h5',
    freqs=high_freqs,
    templates=high_templates,
    metadata=high_metadata
)

print('Saved template_low_freq_noisy_IC.h5')
print('Saved template_high_freq_noisy_IC.h5')