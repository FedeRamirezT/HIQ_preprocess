"""Export all existing desc-cleaning ICA component PSDs (2-45 Hz).

Run with the HIQ_preprocess virtual environment. No ICA is fitted or applied
for cleaning; existing decompositions and EEG inputs are only read.

Place this script in PROJECT/scripts/test/check_IZB_noisy_components.
Run it directly, from any working directory:
    python export_all_ic_psds.py

Output defaults to PROJECT/scripts/test/check_IZB_noisy_components/all_components.h5.
Existing output files are never overwritten. A nonzero exit status indicates
one or more failures; successful recordings remain available in the HDF5.

HDF5 schema (version 1):
  config_json, errors_json: UTF-8 JSON strings
  recordings/000000/: one group per successfully exported recording
    freqs: Hz; psd_mean: (component, frequency), linear power density
    psd_epochs: (epoch, component, frequency), linear power density
    component_index: zero-based index in the saved ICA
    component_names, channel_names, labels_json, excluded_components
    mixing_maps: (channel, component), ICA.get_components(), for later review
    channel_positions: (channel, xyz), metres; NaN if unavailable
    epoch_onsets_seconds: relative to the start of the prepared/cropped Raw
    iclabel_scores, iclabel_score_labels: present only when available
    attributes: recording entities, input paths, processing and PSD parameters

ICA activations have arbitrary scaling: PSD units are activation_unit^2/Hz,
not sensor V^2/Hz. No logarithm, square root or normalization is applied.
Labels are the saved labels, potentially already changed by review_ICs.
The script mirrors the current final-ICA preprocessing configuration; use
the configuration originally used to fit the saved decompositions.
"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys
import traceback

import h5py
import mne
from mne_bids import get_bids_path_from_fname
import numpy as np

TEXT = h5py.string_dtype('utf-8')


def save_text(group, name, value):
    group.create_dataset(name, data=value, dtype=TEXT)


def calculate_psds(ica, raw):
    """Reproduce review_ICs: 4-s epochs, Welch, then mean over epochs."""
    epochs = mne.make_fixed_length_epochs(
        raw, duration=4, overlap=0, preload=True,
        reject_by_annotation=True, verbose=False,
    )
    if len(epochs) == 0:
        raise ValueError('No usable four-second epochs remain.')
    sources = ica.get_sources(epochs)
    n_fft = min(2048, len(sources.times))
    spectrum = sources.compute_psd(
        method='welch', fmin=2, fmax=45, picks='all',
        n_fft=n_fft, n_per_seg=n_fft, n_overlap=n_fft // 2,
        window='hamming', average='mean', remove_dc=True, verbose=False,
    )
    values = spectrum.get_data(picks='all')
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('PSD contains nonfinite or negative values.')
    onsets = (epochs.events[:, 0] - raw.first_samp) / raw.info['sfreq']
    return spectrum.freqs, values, onsets, sources.ch_names, n_fft


def export_recording(config, data_root, ica_path, parent, name):
    from sEEGnal.tools import bids_tools, mne_tools

    bids = get_bids_path_from_fname(ica_path, check=False)
    bids.update(root=data_root, description=None, suffix='eeg',
                extension='.vhdr', check=False)
    ica = bids_tools.read_ica(config, bids, desc='cleaning')
    estimation = config['component_estimation']
    raw = mne_tools.prepare_eeg(
        config, bids, preload=True,
        channels_to_include=config['global']['channels_to_include'],
        channels_to_exclude=config['global']['channels_to_exclude'],
        resample_frequency=estimation['resample_frequency'],
        notch_filter=True,
        freq_limits=[estimation['low_freq'], estimation['high_freq']],
        crop_seconds=estimation['crop_seconds'],
        metadata_badchannels=True, exclude_badchannels=True,
        set_annotations=True, rereference='average',
    )
    try:
        if raw.ch_names != ica.ch_names:
            raise ValueError('Prepared EEG channel order/set differs from fitted ICA. '
                             'Check the original configuration and bad-channel metadata.')
        if raw.info['sfreq'] / 2 < 45 or raw.info['lowpass'] < 45:
            raise ValueError('Prepared data do not cover the requested 2-45 Hz band.')
        freqs, values, onsets, component_names, n_fft = calculate_psds(ica, raw)
        group = parent.create_group(name)
        for entity in ('subject', 'session', 'task', 'run', 'acquisition', 'recording'):
            group.attrs[entity] = getattr(bids, entity, None) or ''
        group.attrs.update(
            ica_path=str(ica_path), eeg_path=str(bids.fpath),
            sfreq=raw.info['sfreq'], raw_first_samp=raw.first_samp,
            prepared_n_times=raw.n_times, n_epochs=len(values),
            n_components=ica.n_components_, ica_method=ica.method,
            epoch_duration_seconds=4, epoch_overlap_seconds=0,
            reject_by_annotation=True, method='welch', window='hamming',
            remove_dc=True, welch_average='mean', epoch_average='mean',
            n_fft=n_fft, n_per_seg=n_fft, n_overlap=n_fft // 2,
            frequency_spacing_hz=raw.info['sfreq'] / n_fft,
            requested_fmin=2, requested_fmax=45,
            psd_units='activation_unit^2/Hz',
        )
        group.create_dataset('freqs', data=freqs)
        group.create_dataset('psd_mean', data=values.mean(axis=0), compression='gzip')
        group.create_dataset('psd_epochs', data=values, compression='gzip', shuffle=True)
        group.create_dataset('component_index', data=np.arange(ica.n_components_))
        group.create_dataset('excluded_components', data=np.array(ica.exclude, dtype=int))
        group.create_dataset('epoch_onsets_seconds', data=onsets)
        group.create_dataset('mixing_maps', data=ica.get_components())
        positions = np.array([channel['loc'][:3] for channel in raw.info['chs']])
        positions[np.all(positions == 0, axis=1)] = np.nan
        group.create_dataset('channel_positions', data=positions)
        save_text(group, 'channel_names', raw.ch_names)
        save_text(group, 'component_names', component_names)
        save_text(group, 'labels_json', json.dumps({
            label: [int(i) for i in indices]
            for label, indices in getattr(ica, 'labels_', {}).items()
        }))
        if hasattr(ica, 'labels_scores_'):
            group.create_dataset('iclabel_scores', data=ica.labels_scores_)
            save_text(group, 'iclabel_score_labels', list(ica.labels_))
    finally:
        raw.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', type=Path,
                        help='Optional override; otherwise inferred from the script location')
    parser.add_argument('--config', type=Path, help='Default: PROJECT/scripts/run/init/config.json')
    parser.add_argument('--output', type=Path, help='New HDF5 file; never overwritten')
    args = parser.parse_args()
    script_folder = Path(__file__).resolve().parent
    project = (args.project_root.expanduser().resolve() if args.project_root
               else script_folder.parents[2])
    # The installed HIQ sEEGnal imports scripts.shared.review_ICs.
    sys.path.insert(0, str(project))
    config_path = (args.config or project / 'scripts/run/init/config.json').resolve()
    with config_path.open(encoding='utf-8') as stream:
        config = json.load(stream)
    data_root = Path(config['path']['data_root']).expanduser()
    if not data_root.is_absolute():
        # Same relative-path convention as scripts/run/init/init.py.
        data_root = project / 'scripts/run' / data_root
    data_root = data_root.resolve()
    config['path']['data_root'] = str(data_root)
    config['path']['sourcedata'] = str(data_root / 'sourcedata/eeg')
    config['path']['recordings'] = str(data_root / 'recordings.tsv')
    config['subsystem'] = 'preprocess'
    input_root = data_root / 'derivatives/sEEGnal/preprocess'
    files = sorted(input_root.rglob('*_desc-cleaning_ica.fif'))
    if not files:
        parser.error(f'No final ICA files found under {input_root}')
    output = (args.output or project / 'scripts/test/check_IZB_noisy_component/all_components.h5').resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    with h5py.File(output, 'x') as h5:
        h5.attrs.update(schema_version=1, status='in_progress',
                        created_utc=datetime.now(timezone.utc).isoformat(),
                        config_path=str(config_path), discovered_recordings=len(files))
        save_text(h5, 'config_json', json.dumps(config, ensure_ascii=False))
        save_text(h5, 'software_versions_json', json.dumps({
            package: version(package) for package in ('mne', 'mne-bids', 'numpy', 'h5py', 'sEEGnal')
        }))
        save_text(h5, 'discovered_ica_files', [str(path) for path in files])
        recordings = h5.create_group('recordings')
        for index, path in enumerate(files):
            name = f'{index:06d}'
            print(f'[{index + 1}/{len(files)}] {path.name}', flush=True)
            try:
                export_recording(config, data_root, path, recordings, name)
            except Exception as error:
                if name in recordings:
                    del recordings[name]
                errors.append(dict(ica_path=str(path), error=str(error),
                                   traceback=traceback.format_exc()))
                print(f'ERROR: {error}', flush=True)
            h5.flush()
        save_text(h5, 'errors_json', json.dumps(errors, ensure_ascii=False))
        h5.attrs['successful_recordings'] = len(recordings)
        h5.attrs['failed_recordings'] = len(errors)
        h5.attrs['status'] = 'complete' if not errors else 'complete_with_errors'
        print(f'Saved {len(recordings)}/{len(files)} recordings to {output}', flush=True)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
