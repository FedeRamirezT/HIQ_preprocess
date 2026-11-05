"""Rank narrow spectral peaks for HUMAN review, without creating templates.

python review_noise_candidates.py --input all_components.h5 --output candidates

Outputs: all_components_ranked.csv (all ICs), candidates.csv (shortlist),
review.html and PNG contact sheets (12 ICs/page). IC indices are ZERO-BASED.
Thresholds are exploratory screening choices, not validated artifact rules.
PSD mean is converted to dB; a 3-Hz running median is the local background.
Candidates have >=2 narrow peaks with >=5 dB prominence and local excess,
or one with >=10 dB prominence and local excess. Width must be <=1 Hz.
Persistence reports the fraction of epochs with >=4 dB excess near the
strongest peak relative to that epoch's own local background.
Labels do not influence selection. All candidates are plotted, with no cap.
"""
import argparse
import csv
import html
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, peak_widths


def db(values):
    return 10 * np.log10(np.maximum(values, np.finfo(float).tiny))


def assess(freqs, psd):
    step = float(np.median(np.diff(freqs)))
    size = max(3, int(round(3 / step)) | 1)
    power_db = db(psd)
    baseline = median_filter(power_db, size=size, mode='nearest')
    peaks, props = find_peaks(power_db, prominence=5)
    widths = peak_widths(power_db, peaks, rel_height=0.5)[0] * step
    excess = power_db[peaks] - baseline[peaks]
    keep = (widths <= 1) & (excess >= 5)
    peaks, prominence, widths, excess = (
        array[keep] for array in (peaks, props['prominences'], widths, excess)
    )
    order = np.argsort(prominence)[::-1]
    peaks, prominence, widths, excess = (
        array[order] for array in (peaks, prominence, widths, excess)
    )
    candidate = len(peaks) >= 2 or bool(np.any((prominence >= 10) & (excess >= 10)))
    score = float(prominence.sum())
    return power_db, baseline, peaks, prominence, widths, excess, candidate, score, size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).with_name('all_components.h5'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('candidates'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    total_recordings = 0
    with h5py.File(args.input, 'r') as h5:
        total_recordings = len(h5['recordings'])
        for key, group in h5['recordings'].items():
            freqs = group['freqs'][:]
            labels = json.loads(group['labels_json'].asstr()[()])
            for index, psd in zip(group['component_index'][:], group['psd_mean'][:]):
                power, background, peaks, prominence, widths, excess, candidate, score, size = assess(freqs, psd)
                row = dict(group=key, subject=str(group.attrs['subject']),
                           session=str(group.attrs['session']), task=str(group.attrs['task']),
                           run=str(group.attrs['run']), component=int(index),
                           labels=';'.join(label for label, values in labels.items() if int(index) in values),
                           candidate=candidate, score=round(score, 3), n_peaks=len(peaks),
                           peak_hz=';'.join(f'{x:.2f}' for x in freqs[peaks]),
                           prominence_db=';'.join(f'{x:.2f}' for x in prominence),
                           width_hz=';'.join(f'{x:.2f}' for x in widths),
                           persistence=0.0, n_epochs=int(group.attrs['n_epochs']), decision='')
                if candidate:
                    epoch_db = db(group['psd_epochs'][:, int(index), :])
                    epoch_excess = epoch_db - median_filter(epoch_db, size=(1, size), mode='nearest')
                    peak = peaks[0]
                    persistence = np.mean(epoch_excess[:, max(0, peak-1):peak+2].max(axis=1) >= 4)
                    row['persistence'] = round(float(persistence), 3)
                rows.append(row)
        rows.sort(key=lambda row: row['score'], reverse=True)
        for rank, row in enumerate(rows, 1):
            row['rank'] = rank
        candidates = [row for row in rows if row['candidate']]
        for filename, selected in [('all_components_ranked.csv', rows), ('candidates.csv', candidates)]:
            with (args.output / filename).open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(selected)
        page_names = []
        for start in range(0, len(candidates), 12):
            fig, axes = plt.subplots(4, 3, figsize=(17, 14), layout='constrained')
            for ax, row in zip(axes.flat, candidates[start:start+12]):
                group = h5['recordings'][row['group']]
                freqs = group['freqs'][:]
                index = row['component']
                power, background, peaks, *_ = assess(freqs, group['psd_mean'][index])
                epochs_db = db(group['psd_epochs'][:, index, :])
                low, high = np.percentile(epochs_db, [10, 90], axis=0)
                ax.fill_between(freqs, low, high, color='steelblue', alpha=.15)
                ax.plot(freqs, power, color='navy', lw=1.3)
                ax.plot(freqs, background, color='gray', ls='--', lw=1)
                ax.scatter(freqs[peaks], power[peaks], color='crimson', s=18)
                for peak in peaks[:5]:
                    ax.annotate(f'{freqs[peak]:g}', (freqs[peak], power[peak]),
                                xytext=(0, 5), textcoords='offset points', fontsize=8)
                ax.set_title(f"#{row['rank']} | sub-{row['subject']} IC {index} | {row['labels']}\n"
                             f"ses-{row['session']} {row['task']} run-{row['run']} | "
                             f"peaks={row['n_peaks']} | persistence={row['persistence']:.0%}", fontsize=10)
                ax.set(xlim=(2,45), xlabel='Frequency (Hz)', ylabel='PSD (dB, arbitrary reference)')
                ax.grid(alpha=.2)
            for ax in list(axes.flat)[len(candidates[start:start+12]):]:
                ax.axis('off')
            fig.suptitle('Candidates for review | zero-based IC indices | red: narrow peaks\n'
                         'Blue: mean PSD; shading: epoch P10-P90; dashed: local background', fontsize=14)
            filename = f'candidates_{start//12+1:02d}.png'
            fig.savefig(args.output / filename, dpi=130)
            plt.close(fig)
            page_names.append(filename)
        content = '<!doctype html><meta charset="utf-8"><title>ICA candidates</title>'
        content += '<style>body{font-family:system-ui;margin:24px}img{width:100%;max-width:1700px}</style>'
        content += f'<h1>{len(candidates)} candidates / {len(rows)} components / {total_recordings} recordings</h1>'
        content += '<p>Exploratory shortlist, not confirmed artifacts. IC indices start at zero. '
        content += 'Decisions can be recorded in the decision column of candidates.csv. '
        content += 'Large physiological peaks may qualify; edge peaks and broader artifacts may be missed.</p>'
        for filename in page_names:
            content += f'<h2>{html.escape(filename)}</h2><a href="{filename}"><img src="{filename}"></a>'
        (args.output / 'review.html').write_text(content, encoding='utf-8')
    print(f'{len(candidates)} candidates / {len(rows)} components / {total_recordings} recordings')
    print('Top candidates:')
    for row in candidates[:24]:
        print(f"#{row['rank']} sub-{row['subject']} IC {row['component']}: {row['peak_hz']} Hz; "
              f"persistence={row['persistence']:.0%}; {row['labels']}")


if __name__ == '__main__':
    main()
