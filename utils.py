import numpy as np
import csv 
def vstack_if_exists(arr1, arr2):
    """
    if arr1 exists: return vstack(arr1,arr2)
    else: return arr2 expanded along first dim
    """

    if type(arr1) == np.ndarray:
        return np.vstack((arr1, arr2))
    else:
        return arr2[None] # expand first dim

def class_data_from_csv(file):
    """
    Inputs:
        file: a csv file, where column 1 contains labels and column 2 contains data
    Returns:
        a dict of {label: data}
    """
    all_data = {}

    # Open the CSV file
    with open(file, 'r') as csvfile:
        # Create a CSV reader
        csvreader = csv.reader(csvfile)
        
        # Skip the header row if it exists
        next(csvreader, None)
        
        # Iterate through the rows and extract the labels (first column)
        for row in csvreader:
            label = row[0]  # Assuming label is in the first column
            data = float(row[1])
            all_data[label] = data

    return all_data