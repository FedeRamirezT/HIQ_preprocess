# Imports
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

def reconstruct_fc_matrix(fc_vector, conn_indices, n_nodes):
    """Reconstruct symmetric FC matrix from upper-triangular vector."""

    matrix = numpy.zeros((n_nodes, n_nodes))

    matrix[conn_indices] = fc_vector
    matrix = matrix + matrix.T

    return matrix


def plot_subject_metric(
    config,
    BIDS,
    metric_function,
    metric_name,
    bands_name,
    ordered_channels
):
    """Plot ordered FC matrices for one subject."""

    # Read one band first to get metadata/channels
    fc, metadata = metric_function(
        config,
        BIDS,
        'sensor',
        band_name=bands_name[0]
    )

    ch_names = list(metadata['ch_names'])
    n_nodes = metadata['n_nodes']

    conn_indices = numpy.triu_indices(n_nodes, k=1)
    idx = [ch_names.index(ch) for ch in ordered_channels]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(15, 10)
    )

    axes = axes.ravel()

    for iband, band_name in enumerate(bands_name):

        fc, metadata = metric_function(
            config,
            BIDS,
            'sensor',
            band_name=band_name
        )

        matrix = reconstruct_fc_matrix(
            fc,
            conn_indices,
            n_nodes
        )

        matrix_ordered = matrix[numpy.ix_(idx, idx)]

        im = axes[iband].imshow(
            matrix_ordered,
            vmin=matrix_ordered.min(),
            vmax=matrix_ordered.max(),
            interpolation='none'
        )

        axes[iband].set_title(band_name)

        axes[iband].set_xticks(range(len(ordered_channels)))
        axes[iband].set_xticklabels(
            ordered_channels,
            rotation=90,
            fontsize=5
        )

        axes[iband].set_yticks(range(len(ordered_channels)))
        axes[iband].set_yticklabels(
            ordered_channels,
            fontsize=5
        )

        # Lines separating LEFT / MIDLINE / RIGHT
        n_left = 26
        n_mid = 6

        axes[iband].axhline(n_left - 0.5, color='k', linewidth=0.8)
        axes[iband].axhline(n_left + n_mid - 0.5, color='k', linewidth=0.8)
        axes[iband].axvline(n_left - 0.5, color='k', linewidth=0.8)
        axes[iband].axvline(n_left + n_mid - 0.5, color='k', linewidth=0.8)

    fig.suptitle(
        metric_name.upper(),
        fontsize=16
    )

    plt.tight_layout()
    plt.show(block=True)
    plt.close(fig)


######################################################################
# Init
######################################################################

config, files, sub, ses, task = init()
config['subsystem'] = 'feature_extraction'

files, sub, ses, task = check_valid(files, sub, ses, task)


######################################################################
# Bands and metrics
######################################################################

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
# Ordered channels: left - midline - right
######################################################################

ordered_channels = [

    # LEFT
    'Fp1',
    'AF7', 'AF3',
    'F7', 'F5', 'F3', 'F1',
    'FT7',
    'FC5', 'FC3', 'FC1',
    'T7',
    'C5', 'C3', 'C1',
    'TP7',
    'CP5', 'CP3', 'CP1',
    'P7', 'P5', 'P3', 'P1',
    'PO7', 'PO3',
    'O1',

    # MIDLINE
    'Fpz',
    'Fz',
    'FCz',
    'Cz',
    'Pz',
    'POz',

    # RIGHT
    'Fp2',
    'AF4', 'AF8',
    'F2', 'F4', 'F6', 'F8',
    'FC2', 'FC4', 'FC6',
    'FT8',
    'C2', 'C4', 'C6',
    'T8',
    'CP2', 'CP4', 'CP6',
    'TP8',
    'P2', 'P4', 'P6', 'P8',
    'PO4', 'PO8',
    'O2'
]


######################################################################
# Main loop
######################################################################

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
        f'\nPlotting subject {current_index}: '
        f'sub {current_sub} '
        f'ses {current_ses} '
        f'task {current_task}\n'
    )

    # First PLV
    plot_subject_metric(
        config=config,
        BIDS=BIDS,
        metric_function=read_plv,
        metric_name=f'PLV - subject {current_index} - {current_sub}',
        bands_name=bands_name,
        ordered_channels=ordered_channels
    )

    # Then ciPLV
    plot_subject_metric(
        config=config,
        BIDS=BIDS,
        metric_function=read_ciplv,
        metric_name=f'ciPLV - subject {current_index} - {current_sub}',
        bands_name=bands_name,
        ordered_channels=ordered_channels
    )