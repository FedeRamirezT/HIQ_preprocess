"""Experiment-specific spectral-line detection for final desc-cleaning ICA.

No templates or output messages. Uses the same 4-s Welch PSD and aperiodic
removal/peak settings as the manually reviewed galleries. Three or more
connected peaks near a 5-Hz grid establish a complete pattern. A partial
5-Hz pair is accepted ONLY when both observed frequencies occur in the
complete pattern of ANOTHER IC of this recording. Partial ICs never provide
support to other ICs. Nonmatches retain their existing categories.

These are exploratory criteria, not a validated noise classifier. Existing
line_noise labels are retained, so run on freshly classified ICLabel output
or use the saved-ICA testing script's default classification reset.
"""
from itertools import combinations

import mne
import numpy as np
from scipy.optimize import linprog
from scipy.signal import correlate, find_peaks, peak_widths
from scipy.sparse import csr_matrix, eye, hstack


SETTINGS = {
    'fmin': 2., 'fmax': 45., 'epoch_seconds': 4.,
    'background_quantile': .25, 'prominence_db': 3., 'height_db': 2.,
    'max_width_hz': 1.5, 'min_peak_distance_hz': .75,
    'nominal_spacing_hz': 5., 'gap_tolerance_hz': .5,
    'alignment_tolerance_hz': .5, 'support_tolerance_hz': .5,
    'peak_map_sigma_hz': .25,
}


def _background(freqs, db, quantile):
    """Compare lower-quantile power-law and exponential fits in dB."""
    fits = []
    n = len(freqs)
    for name, x in [('power_law', np.log10(freqs)), ('exponential', freqs)]:
        design = np.column_stack([np.ones(n), -x])
        objective = np.r_[0., 0., np.full(n, quantile), np.full(n, 1-quantile)]
        result = linprog(objective,
            A_eq=hstack([csr_matrix(design), eye(n), -eye(n)], format='csr'),
            b_eq=db, bounds=[(None, None), (0, None)] + [(0, None)]*(2*n),
            method='highs')
        if not result.success:
            raise RuntimeError('Aperiodic fit failed: '+result.message)
        fits.append((float(result.fun/n), name, design @ result.x[:2]))
    _, name, values = min(fits, key=lambda item: item[0])
    return name, values


def _extract_peaks(freqs, psd, settings):
    if not np.isfinite(psd).all() or np.any(psd < 0) or not np.any(psd > 0):
        return np.array([]), 'invalid_psd'
    db = 10*np.log10(np.maximum(psd, np.finfo(float).tiny))
    name, baseline = _background(freqs, db, settings['background_quantile'])
    residual = db-baseline
    step = float(np.median(np.diff(freqs)))
    peaks, _ = find_peaks(residual, prominence=settings['prominence_db'],
        height=settings['height_db'],
        distance=max(1, int(np.ceil(settings['min_peak_distance_hz']/step))))
    widths = peak_widths(residual, peaks, rel_height=.5)[0]*step
    return freqs[peaks[widths <= settings['max_width_hz']]], name


def _complete_patterns(peaks, settings):
    """Verify connected 5/10-Hz evidence, then fit a grid and retain its peaks.

    Accept triplets with gaps (5,5), (5,10), or (10,5), each within 0.5 Hz.
    Additional peaks need not fit; missing grid positions are allowed.
    """
    nominal = settings['nominal_spacing_hz']
    tol = settings['gap_tolerance_hz']
    patterns = {}
    for indices in combinations(range(len(peaks)), 3):
        values = peaks[list(indices)]
        gaps = np.diff(values)
        for units in [(1, 1), (1, 2), (2, 1)]:
            if np.any(np.abs(gaps-nominal*np.array(units)) > tol):
                continue
            positions = np.array([0, units[0], sum(units)])
            spacing, offset = np.polyfit(positions, values, 1)
            grid_index = np.rint((peaks-offset)/spacing).astype(int)
            errors = np.abs(peaks-(offset+spacing*grid_index))
            # A single grid position may be supported by only one observed line.
            aligned = []
            for k in np.unique(grid_index):
                possible = np.flatnonzero((grid_index == k) &
                    (errors <= settings['alignment_tolerance_hz']))
                if len(possible): aligned.append(int(possible[np.argmin(errors[possible])]))
            if not set(indices).issubset(aligned):
                continue
            aligned = tuple(sorted(aligned))
            info = dict(spacing_hz=float(spacing), offset_hz=float(offset),
                        peaks_hz=peaks[list(aligned)].tolist(),
                        mean_error_hz=float(errors[list(aligned)].mean()))
            if aligned not in patterns or info['mean_error_hz'] < patterns[aligned]['mean_error_hz']:
                patterns[aligned] = info
    return sorted(patterns.values(), key=lambda p: (-len(p['peaks_hz']), p['mean_error_hz']))


def _classify_peak_lists(peak_lists, settings):
    """Two-pass classification independent of saved labels and IC ordering."""
    reports = [dict(component=i, peaks_hz=np.asarray(p).tolist(),
                    complete_patterns=_complete_patterns(np.asarray(p), settings),
                    selected=False, reason='no_supported_pattern', support=[])
               for i, p in enumerate(peak_lists)]
    complete = [r['component'] for r in reports if r['complete_patterns']]
    for i in complete:
        reports[i].update(selected=True, reason='complete_pattern')
    for i, peaks in enumerate(peak_lists):
        if i in complete: continue
        for a, b in combinations(peaks, 2):
            if abs((b-a)-settings['nominal_spacing_hz']) > settings['gap_tolerance_hz']:
                continue
            for j in complete:
                if i == j: continue
                supported = False
                for pattern in reports[j]['complete_patterns']:
                    other = pattern['peaks_hz']
                    # Match actual observed peaks, not inferred empty grid slots.
                    if (any(abs(a-f) <= settings['support_tolerance_hz'] for f in other) and
                            any(abs(b-f) <= settings['support_tolerance_hz'] for f in other)):
                        supported = True
                        break
                if supported:
                    reports[i]['support'].append(dict(component=j, pair_hz=[float(a), float(b)]))
        if reports[i]['support']:
            reports[i].update(selected=True, reason='pair_supported_by_other_complete_ic')
    return reports


def _analyze_psds(psds, freqs, settings=None):
    """Shared analysis of (component, frequency) linear PSDs, for testing."""
    settings = SETTINGS.copy() if settings is None else settings.copy()
    extracted = [_extract_peaks(freqs, psd, settings) for psd in psds]
    reports = _classify_peak_lists([p for p, _ in extracted], settings)
    step = float(np.median(np.diff(freqs)))
    for report, (peaks, model) in zip(reports, extracted):
        report['background_model'] = model
        # Diagnostic only: the actual classification verifies connected peaks.
        if len(peaks):
            peakmap = np.exp(-.5*((freqs[:, None]-peaks[None, :])/
                                  settings['peak_map_sigma_hz'])**2).sum(axis=1)
            ac = correlate(peakmap, peakmap, mode='full', method='direct')[len(freqs)-1:]
            ac /= ac[0]
            maxima, _ = find_peaks(ac, prominence=.04,
                                  distance=max(1, int(np.ceil(1/step))))
            maxima = maxima[(maxima*step >= 1.5) & (maxima*step <= 25)]
            maxima = maxima[np.argsort(ac[maxima])[::-1]][:5]
            report['autocorrelation_lags_hz'] = (maxima*step).tolist()
        else:
            report['autocorrelation_lags_hz'] = []
    return {'method': 'aperiodic_peaks_complete_and_supported_pairs',
            'settings': settings, 'components': reports,
            'selected_components': [r['component'] for r in reports if r['selected']]}


def review_ICs(raw, ica):
    """Reclassify supported spectral patterns as line_noise, silently.

    Nonmatches keep their saved labels; ICLabel probabilities are unchanged.
    Diagnostics are in ica.stable_noise_report_ (in memory, not standard FIF).
    The local test_review_ICs_saved.py script exports this report to JSON.
    """
    with mne.utils.use_log_level('ERROR'):
        epochs = mne.make_fixed_length_epochs(raw, duration=SETTINGS['epoch_seconds'],
            overlap=0, preload=True, reject_by_annotation=True, verbose=False)
        if not len(epochs):
            ica.stable_noise_report_ = {'method': 'aperiodic_peaks_complete_and_supported_pairs',
                'settings': SETTINGS.copy(), 'components': [], 'selected_components': [],
                'reason': 'no_valid_epochs'}
            return ica
        sources = ica.get_sources(epochs)
        n_fft = min(2048, len(sources.times))
        spectrum = sources.compute_psd(method='welch', fmin=SETTINGS['fmin'],
            fmax=SETTINGS['fmax'], picks='all', n_fft=n_fft, n_per_seg=n_fft,
            n_overlap=n_fft//2, window='hamming', average='mean',
            remove_dc=True, verbose=False).average()
        report = _analyze_psds(spectrum.get_data(picks='all'), spectrum.freqs)
    selected = set(report['selected_components'])
    # Only selected ICs leave their previous labels. Unsupported pairs remain.
    if selected:
        for label, indices in ica.labels_.items():
            if label != 'line_noise':
                ica.labels_[label] = [i for i in indices if i not in selected]
        ica.labels_['line_noise'] = sorted(set(ica.labels_.get('line_noise', [])) | selected)
    ica.stable_noise_report_ = report
    return ica
