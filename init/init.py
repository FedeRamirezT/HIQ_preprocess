"""
Prepare the config.json structure

Federico Ramírez-Toraño
28/10/2025
"""

# Imports
import os
import re
import glob
import json


def init():

    # Read the config dictionary
    with open('./init/config.json', 'r') as file:
        config = json.load(file)

    # Folders to find the subjects
    config['path'] = {}
    config['path']['data_root'] = os.path.join('data')
    config['path']['sourcedata'] = os.path.join(config['path']['data_root'], 'sourcedata', 'eeg')

    # Filter the subjects of interest
    files = glob.glob(os.path.join(config['path']['sourcedata'], '*.vhdr'))
    files = [os.path.basename(file) for file in files]

    # Extract the info
    pattern = "HIQ_([0-9]{3})_([0-9])_(1EO|2EC|3EO|4EC|1EC|2WM|3EC)_.*"
    matches = [re.findall(pattern, file) for file in files]
    matches, files = zip(*[(match, file) for match, file in zip(matches, files) if match])
    sub = [match[0][0] for match in matches]
    ses = [match[0][1] for match in matches]
    task = [match[0][2] for match in matches]

    return config, files, sub, ses, task
