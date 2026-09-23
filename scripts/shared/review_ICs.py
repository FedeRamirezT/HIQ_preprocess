"""
Reclassify independent components matching the experiment-specific
low-frequency noise template.

This module belongs to the HIQ preprocessing project and is not part of
the general sEEGnal component-classification logic.
"""

import pathlib

import mne
import h5py
import numpy


def review_ICs(raw, ica):
    """
    Reclassify ICs matching the low-frequency noise template.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording used to obtain the independent-component time series.
        It must contain the same channels used to fit the ICA.
    ica : mne.preprocessing.ICA
        Fitted and already classified ICA object.

    Returns
    -------
    ica : mne.preprocessing.ICA
        The same ICA object, with matching components moved to the
        ``line_noise`` label. ``labels_scores_`` remains unchanged.
    """

    current_folder = pathlib.Path(__file__).resolve().parent
    template_file = (
        current_folder
        / 'templates'
        / 'template_low_freq_noisy_IC.h5'
    )

    with h5py.File(template_file, 'r') as h5:
        template_freqs = h5['freqs'][:]
        templates = h5['templates'][:]

    # Reproduce the four-second epochs used to create the templates.
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=4,
        overlap=0,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )

    # Obtain the independent-component time series for every epoch.
    component_time_series = ica.get_sources(epochs)

    matching_components = set()

    for component in range(ica.n_components_):

        current_component = component_time_series.copy().pick([component])

        # Keep these parameters identical to those used to create the
        # experiment-specific templates.
        n_samples = current_component.get_data().shape[-1]
        n_fft = min(2048, n_samples)

        spectrum = current_component.compute_psd(
            method='welch',
            fmin=2,
            fmax=45,
            picks='all',
            n_fft=n_fft,
            n_per_seg=n_fft,
            n_overlap=n_fft // 2,
        ).average()

        component_freqs = spectrum.freqs
        psd = spectrum.get_data(picks='all')[0]

        # Minor differences in MNE's epoch construction can produce slightly
        # different frequency grids. Resample the current PSD onto the exact
        # grid used by the templates.
        if (
                psd.shape != template_freqs.shape
                or not numpy.allclose(component_freqs, template_freqs)
        ):
            if (
                    template_freqs[0] < component_freqs[0]
                    or template_freqs[-1] > component_freqs[-1]
            ):
                raise ValueError(
                    'The template frequency range is not covered by the '
                    'component PSD. '
                    f'Component range: {component_freqs[0]:.6f}-'
                    f'{component_freqs[-1]:.6f} Hz; '
                    f'template range: {template_freqs[0]:.6f}-'
                    f'{template_freqs[-1]:.6f} Hz.'
                )

            psd = numpy.interp(
                template_freqs,
                component_freqs,
                psd,
            )

        psd = numpy.log10(psd + numpy.finfo(float).eps)
        psd_std = psd.std()

        if psd_std == 0:
            continue

        psd = (psd - psd.mean()) / psd_std

        for template in templates:
            correlation = numpy.corrcoef(template, psd)[0, 1]

            if correlation > 0.8:
                matching_components.add(component)
                break

    # Move every match out of its previous category. Iterating through all
    # existing categories also handles any additional labels created by MNE.
    for component in matching_components:
        for label, labeled_components in ica.labels_.items():
            if label != 'line_noise' and component in labeled_components:
                labeled_components.remove(component)

        if component not in ica.labels_.setdefault('line_noise', []):
            ica.labels_['line_noise'].append(component)

    ica.labels_['line_noise'].sort()

    return ica


# High-frequency template matching is intentionally disabled for now.
# Its future implementation should load:
#
# template_high_freq_noisy_IC.h5
#
# and reproduce the original 60–100 Hz comparison.