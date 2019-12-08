#!/usr/bin/python3

"""
This file captures all of the folder names in data-2019/test/ and stores them
in a csv file named test_results.csv.  Each of these folders then get assigned
a random label (0-2), as defined in the "labels" column (last column) of
classes.csv.
"""

import os # operating system module
import csv # alternate CSV file I/O tool (built into Python already)
import random # randomly generate labels for now


def generate_output_csv(labels):
    """Puts test data folder names into csv folder, along with a random label.

    Input arguments:

        labels: A vector containing the NN output for the test data. Each index is 0-3

    Output arguments:

        None

    """
    # Test data files are available in the data-2019/test folder
    test_data_dir = os.path.abspath(__file__).replace('generate_output_csv.py', 'data-2019/test')

    # Set up csv rows list and specify filename in same directory as generate_output_csv.py
    csv_rows = [['guid/image', 'label']]
    csv_filename = os.path.abspath(__file__).replace('generate_output_csv.py', 'test_results.csv')

    # Step through all files in test_data_dir
    index = 0
    for dirname, _, filenames in os.walk(test_data_dir):
        for filename in filenames:
            # Only care about the .jpg files
            if filename[len(filename) - 3: len(filename)] == 'jpg':
                # 1st four chars of filename are the file number
                file_num = filename[0:4]
                # Last list item of split dirname is the folder number
                folder_num = dirname.split('/')[len(dirname.split('/')) - 1]
                # Join folder number and file number to produce guid
                guid = os.path.join(folder_num, file_num)
    #             print(guid) # optionally print guid's to check if this is working
                # Append new row to csv_rows list
                csv_rows.append([guid, labels(index)])
                index = index + 1

    # Any results you write to the current directory are saved as output.

    # Write rows list to csv file
    with open(csv_filename, 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(csv_rows)

    print(len(csv_rows), ' rows written to the ', csv_filename, ' csv file.')