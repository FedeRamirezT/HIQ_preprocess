"""
Script to extract features automatically

Federico Ramírez-Toraño
24/02/2026

"""

# Imports
from init.init import init
from scripts.shared.check_valid import check_valid
from sEEGnal.tools.bids_tools import build_BIDS_object
from sEEGnal.feature_extraction.feature_extraction import feature_extraction


# Init the database
config, files, sub, ses, task = init()

# Keep only the valid ones
files, sub, ses, task = check_valid(files, sub, ses, task)

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

    # Run the feature extraction
    print('   Feature extraction.')
    results = feature_extraction(config,BIDS)
