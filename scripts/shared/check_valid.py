"""
Read the QC Excel file and decide if valid or not

Federico Ramírez-Toraño
09/06/2026

"""

import os
import pandas
import numpy


def check_valid(files, sub, ses, task):

    # If not QC file, all valid
    qc_file = os.path.join('excels','QC.xlsx')

    if not os.path.isfile(qc_file):
        return files, sub, ses, task

    # Open QC file
    qc_data = pandas.read_excel(qc_file)

    # Create all valid mask
    qc_mask = numpy.ones(len(files), dtype=bool)

    # For each file
    for ifile,current_file in enumerate(files):

        # Look for the file in the Excel
        current_file = current_file[:-5]
        index = [i for i in range(len(qc_data)) if current_file == qc_data['archivo'][i]]

        # Check if valid
        if len(index) == 1:

            index = index[0]
            if qc_data['Valid'][index] == 0:
                qc_mask[ifile] = False

        else:
            qc_mask[ifile] = False

    # Apply mask to each variable
    files = [f for f, m in zip(files, qc_mask) if m]
    sub = [s for s, m in zip(sub, qc_mask) if m]
    ses = [s for s, m in zip(ses, qc_mask) if m]
    task = [t for t, m in zip(task, qc_mask) if m]
    return files, sub, ses, task
