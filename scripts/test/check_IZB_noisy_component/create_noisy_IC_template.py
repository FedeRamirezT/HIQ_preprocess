"""Create 2-45 Hz templates from manually selected saved desc-cleaning ICs.

Uses the same preprocessing and Welch/log10/z-score calculation as
plot_IC_psd.py and template-based review_ICs.py. No ICA is fitted.
Indices are zero-based indices of the saved ICA. One spectrum is retained
per selected IC; spectra are NOT averaged across components or subjects.

Default output: scripts/shared/templates/template_low_freq_noisy_IC.h5,
the file loaded by review_ICs. Existing output is replaced only after all
selected ICs have been successfully processed. --output changes that path.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import h5py
import numpy


# Keep synchronized with indices selected in the current plot_IC_psd figures.
low_freq_noisy_IC = {
    '024': [21],
    '028': [8],
    '029': [8],
    '031': [7],
    '050': [11],
    '051': [6],
    '087': [28],
    '092': [10],
    '095': [7],
    '096': [9],
    '103': [22],
    '107': [8],
    '111': [12],
    '113': [7],
    '119': [11],
}


def compute_ic_psd_template(config, recordings, noisy_ic_dict, fmin=2, fmax=45):
    import mne
    from sEEGnal.tools import bids_tools, mne_tools

    chosen = [r for r in recordings if r['subject'] in noisy_ic_dict]
    missing = set(noisy_ic_dict) - {r['subject'] for r in chosen}
    if missing:
        raise ValueError('Selected subjects absent from manifest: '+', '.join(sorted(missing)))
    if not chosen or any(not indices for indices in noisy_ic_dict.values()):
        raise ValueError('Select at least one IC per listed subject')
    templates, metadata, runs, provenance = [], [], [], []
    freqs_template = None
    ce = config['component_estimation']
    for recording in chosen:
        bids = recording['bids_path']
        subject = recording['subject']
        indices = noisy_ic_dict[subject]
        if len(set(indices)) != len(indices):
            raise ValueError(f'Duplicate selected IC indices in sub-{subject}')
        print(f"sub-{subject} ses-{recording['session']} task-{recording['task']} "
              f"run-{recording['run']}: ICs {indices}", flush=True)
        raw = None
        try:
            with mne.utils.use_log_level('ERROR'):
                ica = bids_tools.read_ica(config, bids, desc='cleaning')
                if any(type(i) is not int or i < 0 or i >= ica.n_components_ for i in indices):
                    raise ValueError(f'sub-{subject}: IC indices must be integers in 0..{ica.n_components_-1}')
                raw = mne_tools.prepare_eeg(
                    config, bids, preload=True,
                    channels_to_include=config['global']['channels_to_include'],
                    channels_to_exclude=config['global']['channels_to_exclude'],
                    notch_filter=True, freq_limits=[ce['low_freq'], ce['high_freq']],
                    resample_frequency=ce['resample_frequency'],
                    metadata_badchannels=True, exclude_badchannels=True,
                    interpolate_badchannels=False, set_annotations=True,
                    crop_seconds=ce['crop_seconds'], rereference='average',
                )
                if raw.ch_names != ica.ch_names:
                    raise ValueError(f'sub-{subject}: prepared channels/order differ from fitted ICA')
                epochs = mne.make_fixed_length_epochs(raw, duration=4, overlap=0,
                    preload=True, reject_by_annotation=True, verbose=False)
                if not len(epochs):
                    raise ValueError(f'sub-{subject}: no valid four-second epochs')
                sources = ica.get_sources(epochs)
                n_fft = min(2048, len(sources.times))
                spectrum = sources.compute_psd(
                    method='welch', fmin=fmin, fmax=fmax, picks='all',
                    n_fft=n_fft, n_per_seg=n_fft, n_overlap=n_fft//2,
                    window='hamming', average='mean', remove_dc=True,
                    verbose=False,
                ).average()
                psds = spectrum.get_data(picks='all')
                freqs = spectrum.freqs
            if freqs_template is None:
                freqs_template = freqs.copy()
            elif not numpy.array_equal(freqs_template, freqs):
                raise ValueError(f'sub-{subject}: frequency grid differs from previous references')
            ica_path = Path(bids_tools.build_derivatives_path(bids, 'preprocess', 'desc-cleaning_ica.fif'))
            digest = hashlib.sha256(ica_path.read_bytes()).hexdigest()
            for component in indices:
                psd = psds[component]
                if not numpy.isfinite(psd).all() or (psd < 0).any():
                    raise ValueError(f'sub-{subject} IC {component}: invalid PSD')
                psd = numpy.log10(psd + numpy.finfo(float).eps)
                if not numpy.isfinite(psd).all() or psd.std() == 0:
                    raise ValueError(f'sub-{subject} IC {component}: constant/invalid log PSD')
                templates.append((psd-psd.mean())/psd.std())
                metadata.append((subject,recording['session'],recording['task'],str(component)))
                runs.append(recording['run'])
                provenance.append(dict(subject=subject,session=recording['session'],
                    task=recording['task'],run=recording['run'],component=component,
                    ica_path=str(ica_path),ica_sha256=digest,n_epochs=len(epochs),
                    sfreq=raw.info['sfreq'],n_fft=n_fft,n_per_seg=n_fft,n_overlap=n_fft//2))
        finally:
            if raw is not None:
                raw.close()
    return freqs_template, numpy.asarray(templates), numpy.asarray(metadata,dtype='S'), runs, provenance


def save_template_h5(output_file, freqs, templates, metadata, runs, provenance, config):
    output_file = Path(output_file).resolve()
    output_file.parent.mkdir(parents=True,exist_ok=True)
    # Write a temporary file in the same directory, then atomically replace.
    fd, temporary = tempfile.mkstemp(prefix='.noise_template_',suffix='.h5',dir=output_file.parent)
    os.close(fd)
    try:
        with h5py.File(temporary,'w') as h5:
            h5.create_dataset('freqs',data=freqs)
            h5.create_dataset('templates',data=templates)
            h5.create_dataset('metadata',data=metadata)
            text = h5py.string_dtype('utf-8')
            h5.create_dataset('run',data=[str(r) for r in runs],dtype=text)
            h5.create_dataset('provenance_json',data=json.dumps(provenance),dtype=text)
            h5.create_dataset('config_json',data=json.dumps(config),dtype=text)
            h5.attrs.update(created_utc=datetime.now(timezone.utc).isoformat(),
                component_index_base=0,ica_description='cleaning',epoch_duration_seconds=4,
                epoch_overlap_seconds=0,reject_by_annotation=True,method='welch',
                window='hamming',remove_dc=True,epoch_average='mean',
                normalization='log10(psd + float64 eps), then z-score ddof=0')
        os.replace(temporary,output_file)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=folder.parents[2])
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    sys.path.insert(0,str(root))
    from scripts.run.init.init import init
    config, recordings = init()
    config['subsystem'] = 'preprocess'
    result = compute_ic_psd_template(config,recordings,low_freq_noisy_IC)
    output = args.output or root/'scripts/shared/templates/template_low_freq_noisy_IC.h5'
    save_template_h5(output,*result,config)
    print(f'Saved {len(result[1])} spectra, {len(result[0])} frequencies: {Path(output).resolve()}')


if __name__ == '__main__':
    main()
