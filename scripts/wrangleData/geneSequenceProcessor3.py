import os
import pandas as pd


"""
Load data
"""
path_data = "C:/Users/HP/PycharmProjects/Zattention/data/prep 4.xlsx"
path_data_Folder = "C:/Users/HP/PycharmProjects/Zattention/data/"
data = pd.read_excel(path_data, header=None)

"""
Function to add spaces between nucleotides such that each nucleotide
    can later be considered as a word
"""


def add_spaces(sequence):
    return " ".join(sequence)


"""
Apply the function to both columns
"""
print(data[0])
data[0] = data[0].apply(add_spaces)
data[1] = data[1].apply(add_spaces)

"""
Save the modified DataFrame to a new Excel file
"""
# EXCEL
data.to_excel(
    os.path.join(path_data_Folder, 'processed 1.xlsx'),
    index=False, header=False)
