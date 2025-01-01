import re
import os
import pandas as pd


"""
Load data
"""
path_data = "C:/Users/HP/PycharmProjects/Zattention/data/prep 2.xlsx"
path_data_Folder = "C:/Users/HP/PycharmProjects/Zattention/data/"
data = pd.read_excel(path_data, header=None)

"""
Function to extract identifier code from a sequence identifier line 
"""


def extract_code_from_gene(identifier):
    identifier = str(identifier)
    match = re.search(r'>(\w+\.\d+)', identifier)
    if match:
        return match.group(1)
    return None


def extract_code_from_cds(identifier):
    identifier = str(identifier)
    match = re.search(r'lcl\|(\w+\.\d+)', identifier)
    if match:
        return match.group(1)
    return None


"""
Create dictionaries to store sequences by their identifier codes 
"""
gene_dict = {}
coding_dict = {}

"""
Populate the dictionaries with sequences from the DataFrame 
"""
for i in range(0, len(data), 2):
    gene_id = extract_code_from_gene(data.iloc[i, 0])
    gene_seq = data.iloc[i + 1, 0]
    gene_dict[gene_id] = (data.iloc[i, 0], gene_seq)

    coding_id = extract_code_from_cds(data.iloc[i, 1])
    coding_seq = data.iloc[i + 1, 1]
    if coding_id not in coding_dict:
        coding_dict[coding_id] = (data.iloc[i, 1], coding_seq)

"""
Create lists to store the matched sequences
"""
matched_genes = []
matched_codings = []

"""
# Match gene sequences with their corresponding coding sequences 
"""
for gene_id in gene_dict:
    if gene_id in coding_dict:
        matched_genes.append(gene_dict[gene_id])
        matched_codings.append(coding_dict[gene_id])

"""
Create a new DataFrame with the matched sequences
"""
matched_df = pd.DataFrame({
    'Gene Identifier': [item[0] for item in matched_genes],
    'Gene Sequence': [item[1] for item in matched_genes],
    'Coding Identifier': [item[0] for item in matched_codings],
    'Coding Sequence': [item[1] for item in matched_codings] })

"""
Save the matched data to a new Excel file 
"""
matched_df.to_excel(
    os.path.join(path_data_Folder, 'prep 3.xlsx'),
    index=False, header=False)
