import os
import pandas as pd


"""
Load data
"""
path_data = "C:/Users/HP/PycharmProjects/Zattention/data/prep 1.xlsx"
path_data_Folder = "C:/Users/HP/PycharmProjects/Zattention/data/"
data = pd.read_excel(path_data, header=None)

"""
Function to process a column of sequences
"""


def process_column(column):
    processed_data = []
    current_sequence = ""

    for row in column:
        if pd.isna(row):
            continue
        if row.startswith(">"):
            if current_sequence:
                processed_data.append(current_sequence)
            processed_data.append(row)
            current_sequence = ""
        else:
            current_sequence += row.strip()
    if current_sequence:
        processed_data.append(current_sequence)
    return processed_data


"""
Process both columns
"""
processed_column_1 = process_column(data[0])
processed_column_2 = process_column(data[1])

"""
Create a new DataFrame with the processed data and save it
    Ensure both columns have the same length by padding the 
    shorter one with empty strings 
"""
max_length = max(len(processed_column_1), len(processed_column_2))
processed_column_1.extend([''] * (max_length - len(processed_column_1)))
processed_column_2.extend([''] * (max_length - len(processed_column_2)))

dictionary = {
    "Genes": processed_column_1,
    "Coding Sequences": processed_column_2}
processed_dataFrame = pd.DataFrame(dictionary)

processed_dataFrame.to_excel(
    os.path.join(path_data_Folder, 'prep 2.xlsx'),
    index=False, header=False)
