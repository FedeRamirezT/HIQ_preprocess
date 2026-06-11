# Imports
import os

import matplotlib.pyplot as plt

from init.init import init
from sEEGnal.tools.bids_tools import build_BIDS_object
import sEEGnal.tools.bids_tools as bids_tools

# Fix qtagg
import matplotlib
matplotlib.use('tkagg')


# Init the database
config, files, sub, ses, task = init()
config['subsystem'] = 'preprocess'

# Output
badchannels = []

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

    # Load bad channels
    chan = bids_tools.read_badchannels(config, BIDS)
    current_badchannels = list(chan.loc[chan['status'] == 'bad']['name'])

    # Save
    badchannels.append(len(current_badchannels))


# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
ax.plot(badchannels,'-*')

ax. axhline(y=7, color='r')
ax.grid()

ax.set_xlabel("Subject")
ax.set_ylabel("Number of bad channels")
plt.tight_layout()

figure_path = os.path.join('scripts','test','summary_presentation','figures','number_of_bad_channels.png')
plt.savefig(figure_path)
plt.close()