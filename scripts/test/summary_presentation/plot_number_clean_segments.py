# Imports
import os

import matplotlib.pyplot as plt

from init.init import init
from sEEGnal.tools.bids_tools import build_BIDS_object
from sEEGnal.tools.mne_tools import prepare_eeg
# Fix qtagg
import matplotlib
matplotlib.use('tkagg')


# Init the database
config, files, sub, ses, task = init()
config['subsystem'] = 'preprocess'

# Output
clean_segments = []

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

    # Load the clean EEG
    sobi = {
        'desc': 'sobi',
        'components_to_include': ['brain', 'other'],
        'components_to_exclude': []
    }

    freq_limits = [
        config['component_estimation']['low_freq'],
        config['component_estimation']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    epoch_definition = config['source_reconstruction']['epoch_definition']

    # Load the clean data
    raw = prepare_eeg(
        config,
        BIDS,
        preload=True,
        crop_seconds=crop_seconds,
        resample_frequency=resample_frequency,
        set_annotations=True,
        epoch_definition=epoch_definition
    )

    # Save
    clean_segments.append(raw.events.shape[0])


# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
ax.plot(clean_segments,'-*')

ax. axhline(y=20, color='r')
ax.grid()

ax.set_xlabel("Subject")
ax.set_ylabel("Number of clean segments")
plt.tight_layout()

figure_path = os.path.join('scripts','test','summary_presentation','figures','number_of_clean_segments.png')
plt.savefig(figure_path)
plt.close()