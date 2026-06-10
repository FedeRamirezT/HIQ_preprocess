"""
Script to call automatic preprocess for all EEGs

Federico Ramírez-Toraño
28/10/2025

"""

# Imports
import os
from dev.init.init import init
from sEEGnal.tools.bids_tools import build_BIDS_object
import sEEGnal.tools.bids_tools as bids
import matplotlib.pyplot as plt
import numpy
import pandas

# Fix plot
import matplotlib
matplotlib.use('Qt5Agg')

# Init the database
config, files, sub, ses, task = init()

# Output folder for figures
config['path']['correlation'] = os.path.join('validation','check_IZB_noisy_component','components_figures','correlation')

# List of subjects with errors
errors = []

# Go through each subject
index = range(len(files))
for current_index in index:

    # current info
    current_file = files[current_index]
    current_sub = sub[current_index]
    current_ses = ses[current_index]
    current_task = task[current_index]

    # Create the subjects following AI-Mind protocol
    BIDS = build_BIDS_object(config, current_sub, current_ses, current_task)

    print('Working with sub ' + current_sub + ' ses ' + current_ses + ' task ' + current_task)

    # Load mixing matrix
    config['subsystem'] = 'preprocess'
    fname_mixing = bids.build_derivatives_path(BIDS, config['subsystem'], 'desc-sobi_mixing.tsv')
    mixing = pandas.read_csv(fname_mixing, sep="\t")

    # Get the info
    channels = mixing["output_name"].to_list()
    ic_names = mixing.columns[1:].to_list()
    A = mixing[ic_names].to_numpy(float)

    # Get correlation
    corr_ics = numpy.abs(numpy.corrcoef(A.T))

    # Plot
    plt.subplot(1,2,1)
    for i in range(A.shape[1]):
        plt.plot(A[:,i])

    plt.subplot(1, 2, 2)
    plt.imshow(corr_ics, vmin=0, vmax=1)
    plt.colorbar(label="abs spatial correlation")

    # Save fig
    output_file = os.path.join(
        config['path']['correlation'],
        f"{current_sub}_ICs_correlation.png"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
