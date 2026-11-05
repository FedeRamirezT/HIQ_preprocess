"""Plot all saved final-ICA components individually, twelve per figure.

Edit RECORDING_INDICES or run --subjects 091 110 121 / --indices 8 9.
Uses the current init(), desc-cleaning ICA, and review_ICs PSD calculation.
Displays z-scored log10 PSDs, without templates, correlations or selection.
PNG pages and an HTML gallery are saved; --no-show suppresses windows.
All IC indices are zero-based indices of the saved ICA, not SOBI indices.
"""
import argparse
from datetime import datetime
import html
from pathlib import Path
import sys

import numpy as np


RECORDING_INDICES = [72, 73, 83, 87,  89, 97, 98, 99, 100, 101]


def normalize(values):
    values = np.log10(values + np.finfo(float).eps)
    scale = values.std()
    if not np.isfinite(values).all() or scale == 0:
        return None
    return (values - values.mean()) / scale


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', type=Path, default=folder.parents[2])
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--subjects', nargs='+', help='Subject IDs, e.g. 091 110')
    selection.add_argument('--indices', nargs='+', type=int, help='Zero-based manifest row indices')
    parser.add_argument('--output', type=Path,
                        default=folder / ('ic_psd_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    parser.add_argument('--no-show', action='store_true', help='Save plots without opening windows')
    args = parser.parse_args()
    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    else:
        matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import mne
    root = args.project_root.expanduser().resolve()
    sys.path.insert(0, str(root))
    from scripts.run.init.init import init
    from sEEGnal.tools import bids_tools, mne_tools

    config, recordings = init()
    config['subsystem'] = 'preprocess'
    if args.subjects:
        wanted = {s.removeprefix('sub-') for s in args.subjects}
        selected = [r for r in recordings if r['subject'] in wanted]
        missing = wanted - {r['subject'] for r in selected}
        if missing:
            parser.error('Subjects not in manifest: ' + ', '.join(sorted(missing)))
    else:
        indices = args.indices if args.indices is not None else RECORDING_INDICES
        if any(i < 0 or i >= len(recordings) for i in indices):
            parser.error(f'Recording indices must be in 0..{len(recordings)-1}')
        selected = [recordings[i] for i in dict.fromkeys(indices)]
    if not selected:
        parser.error('No recordings selected')
    args.output.mkdir(parents=True, exist_ok=False)
    gallery = []
    for recording in selected:
        bids = recording['bids_path']
        stem = bids.basename.removesuffix('_eeg.vhdr').removesuffix('_eeg')
        raw = None
        try:
            with mne.utils.use_log_level('ERROR'):
                ica = bids_tools.read_ica(config, bids, desc='cleaning')
                ce = config['component_estimation']
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
                    raise ValueError('Prepared channel names/order differ from the fitted ICA')
                epochs = mne.make_fixed_length_epochs(raw, duration=4, overlap=0,
                    preload=True, reject_by_annotation=True, verbose=False)
                if not len(epochs):
                    raise ValueError('No valid four-second epochs')
                sources = ica.get_sources(epochs)
                n_fft = min(2048, len(sources.times))
                spectrum = sources.compute_psd(method='welch', fmin=2, fmax=45,
                    picks='all', n_fft=n_fft, n_per_seg=n_fft, n_overlap=n_fft//2,
                    window='hamming', average='mean', remove_dc=True,
                    verbose=False).average()
                psds = spectrum.get_data(picks='all')
                freqs = spectrum.freqs
            normalized = [normalize(psd) for psd in psds]
            title = (f'{stem} | desc-cleaning | {ica.n_components_} ICs | '
                     f'{len(epochs)} valid epochs | zero-based ICA indices')
            pages = []
            for start in range(0, ica.n_components_, 12):
                fig, axes = plt.subplots(4, 3, figsize=(17, 13), layout='constrained')
                manager = plt.get_current_fig_manager()
                manager.window.state('zoomed')
                batch = list(range(start, min(start + 12, ica.n_components_)))
                for ax, component in zip(axes.flat, batch):
                    current = normalized[component]
                    if current is not None:
                        ax.plot(freqs, current, color='navy', lw=1.3)
                    else:
                        ax.text(.5, .5, 'Constant or invalid PSD', transform=ax.transAxes,
                                ha='center', va='center')
                    labels = '; '.join(k for k, v in ica.labels_.items() if component in v)
                    ax.set_title(f"sub-{recording['subject']} | IC {component} | {labels or 'unlabeled'}\n"
                                 f"ses-{recording['session']} | task-{recording['task']} | run-{recording['run']}",
                                 fontsize=9)
                    ax.set(xlim=(freqs[0], freqs[-1]), xlabel='Frequency (Hz)',
                           ylabel='z-score of log10 PSD')
                    ax.grid(alpha=.2)
                for ax in list(axes.flat)[len(batch):]:
                    ax.axis('off')
                fig.suptitle(title, fontsize=12)
                filename = f'{stem}_ICs_{start:03d}_{batch[-1]:03d}.png'
                fig.savefig(args.output / filename, dpi=120)
                pages.append(filename)
                if not args.no_show:
                    plt.show(block=True)
                plt.close(fig)
            gallery.append((stem, pages))
        finally:
            if raw is not None: raw.close()
    content = '<!doctype html><meta charset="utf-8"><title>ICA PSD review</title><style>body{font-family:system-ui;margin:24px}img{width:100%;max-width:1800px}</style>'
    content += '<h1>Saved desc-cleaning ICA spectra</h1><p>IC indices start at zero. Each panel shows one component. Saved labels are context only and are not modified.</p>'
    for stem, pages in gallery:
        content += '<h2>'+html.escape(stem)+'</h2>'
        for name in pages:
            escaped = html.escape(name, quote=True)
            content += f'<a href="{escaped}"><img loading="lazy" src="{escaped}"></a>'
    (args.output / 'review.html').write_text(content, encoding='utf-8')
    print('Gallery: '+str((args.output / 'review.html').resolve()))


if __name__ == '__main__':
    main()
