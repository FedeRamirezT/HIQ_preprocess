# Imports
import os

import numpy
import matplotlib.pyplot as plt

from init.init import init
from sEEGnal.tools.bids_tools import (
    build_BIDS_object,
    read_plv,
    read_ciplv
)
from scripts.shared.check_valid import check_valid

# Fix qtagg
import matplotlib
matplotlib.use('tkagg')


######################################################################
# Functions
######################################################################

def fc_vector_to_strength(fc_vector, n_nodes):
    """
    Convert upper-triangular FC vector to nodal strength.

    fc_vector must be the upper triangle obtained with:
        numpy.triu_indices(n_nodes, k=1)
    """

    fc_matrix = numpy.zeros((n_nodes, n_nodes))

    triu_idx = numpy.triu_indices(n_nodes, k=1)

    fc_matrix[triu_idx] = fc_vector
    fc_matrix = fc_matrix + fc_matrix.T

    strength = numpy.sum(fc_matrix, axis=1)

    return strength


def load_metric_band_strength(
    config,
    files,
    sub,
    ses,
    task,
    metric_function,
    band_name
):
    """Load one connectivity metric and one frequency band as nodal strength."""

    strength_all = []

    for current_index in range(len(files)):

        current_sub = sub[current_index]
        current_ses = ses[current_index]
        current_task = task[current_index]

        BIDS = build_BIDS_object(
            config,
            current_sub,
            current_ses,
            current_task
        )

        print(
            f'Working with sub {current_sub} '
            f'ses {current_ses} '
            f'task {current_task} '
            f'band {band_name}'
        )

        fc_vector, metadata = metric_function(
            config,
            BIDS,
            'sensor',
            band_name=band_name
        )

        n_nodes = metadata['n_nodes']

        strength = fc_vector_to_strength(
            fc_vector,
            n_nodes
        )

        strength_all.append(strength)

    return numpy.asarray(strength_all)


######################################################################
# Init
######################################################################

config, files, sub, ses, task = init()
config['subsystem'] = 'feature_extraction'

files, sub, ses, task = check_valid(files, sub, ses, task)

figures_dir = os.path.join(
    'scripts',
    'test',
    'summary_presentation',
    'figures'
)

os.makedirs(figures_dir, exist_ok=True)

bands_name = [
    'delta',
    'theta',
    'alpha',
    'low_beta',
    'high_beta',
    'gamma'
]

metrics = {
    'plv': read_plv,
    'ciplv': read_ciplv
}


######################################################################
# Main loop
######################################################################

for metric_name, metric_function in metrics.items():

    print(f'\nProcessing {metric_name.upper()} STRENGTH\n')

    strength_by_band = {}

    for band_name in bands_name:

        strength_by_band[band_name] = load_metric_band_strength(
            config,
            files,
            sub,
            ses,
            task,
            metric_function,
            band_name
        )

    ##################################################################
    # 1. Strength matrices by band
    ##################################################################

    fig, axes = plt.subplots(
        nrows=len(bands_name),
        ncols=1,
        figsize=(14, 14),
        sharex=True,
        sharey=True
    )

    for iband, band_name in enumerate(bands_name):

        strength_all = strength_by_band[band_name]

        im = axes[iband].imshow(
            strength_all,
            aspect='auto',
            interpolation='none',
            vmin=numpy.nanmin(strength_all),
            vmax=numpy.nanmax(strength_all)
        )

        axes[iband].set_ylabel('Subject')
        axes[iband].set_title(
            f'{metric_name.upper()} strength - {band_name}'
        )

        fig.colorbar(
            im,
            ax=axes[iband],
            label='Strength'
        )

    axes[-1].set_xlabel('Node / channel')

    plt.tight_layout()

    figure_path = os.path.join(
        figures_dir,
        f'strength_all_{metric_name}_bands.png'
    )

    plt.savefig(figure_path, dpi=300)
    plt.close()

    ##################################################################
    # 2. Subject-by-subject correlation based on strength
    ##################################################################

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(15, 9)
    )

    axes = axes.ravel()

    for iband, band_name in enumerate(bands_name):

        strength_all = strength_by_band[band_name]

        r_all = numpy.corrcoef(strength_all)

        im = axes[iband].imshow(
            r_all,
            vmin=-1,
            vmax=1,
            cmap='bwr'
        )

        axes[iband].set_title(band_name)
        axes[iband].set_ylabel('Subject')

    fig.suptitle(
        f'{metric_name.upper()} strength subject-by-subject correlation',
        fontsize=14
    )

    fig.colorbar(
        im,
        ax=axes,
        label='Correlation'
    )

    plt.tight_layout()

    figure_path = os.path.join(
        figures_dir,
        f'strength_corr_all_{metric_name}_bands.png'
    )

    plt.savefig(figure_path, dpi=300)
    plt.close()

    ##################################################################
    # 3. Strength distribution per subject by band
    ##################################################################

    fig, axes = plt.subplots(
        nrows=len(bands_name),
        ncols=1,
        figsize=(18, 18),
        sharex=True
    )

    for iband, band_name in enumerate(bands_name):

        strength_all = strength_by_band[band_name]
        ax = axes[iband]

        vmin = numpy.nanmin(strength_all)
        vmax = numpy.nanmax(strength_all)

        for isub in range(strength_all.shape[0]):

            x = isub + 0.15 * (
                numpy.random.rand(strength_all.shape[1]) - 0.5
            )

            ax.scatter(
                x,
                strength_all[isub, :],
                alpha=0.7,
                s=20,
                color='tab:blue'
            )

            ax.scatter(
                isub,
                numpy.nanmean(strength_all[isub, :]),
                alpha=1,
                s=80,
                color='crimson',
                edgecolor='black',
                linewidth=0.5,
                zorder=10
            )

        ax.set_ylabel('Strength')
        ax.set_title(f'{band_name}')
        ax.set_ylim(vmin, vmax)

    axes[-1].set_xlabel('Subject')
    axes[-1].set_xticks(numpy.arange(strength_all.shape[0]))
    axes[-1].set_xticklabels(
        numpy.arange(strength_all.shape[0]),
        rotation=90
    )

    fig.suptitle(
        f'{metric_name.upper()} strength distribution per subject',
        fontsize=14
    )

    plt.tight_layout()

    figure_path = os.path.join(
        figures_dir,
        f'{metric_name}_strength_distribution_per_subject_bands.png'
    )

    plt.savefig(figure_path, dpi=300)
    plt.close()