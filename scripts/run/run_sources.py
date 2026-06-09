"""
Scripts to call sources_reconstruction

Federico Ramírez-Toraño
11/02/2026

"""

# Imports
from dev.init.init import init
from sEEGnal.tools.bids_tools import build_BIDS_object
from sEEGnal.sources_reconstruction.forward import make_forward_model
from sEEGnal.sources_reconstruction.inverse import estimate_inverse_solution

# What step to run: forward, inverse
run = [1, 1]

# Init the database
config, files, sub, ses, task = init()

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
    print(current_file)

    # Create the subjects following AI-Mind protocol
    BIDS = build_BIDS_object(config, current_sub, current_ses, current_task)

    # Run the selected processes
    if run[0]:
        print('   Forward Model', end='. ')
        results = make_forward_model(config, BIDS)
        print(' Result ' + results['result'])

    if run[1]:
        print('   Inverse Solution', end='. ')
        results = estimate_inverse_solution(config, BIDS)
        print(' Result ' + results['result'])
