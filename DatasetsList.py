import os
import pandas as pd



def get_datasets_list(threshold, root_dir):

    # Only keep desc of Datasets with fixed length
    data_summary = pd.read_csv('DataSummary.csv')
    data_summary['Length'] = pd.to_numeric(data_summary['Length'], errors='coerce')
    data_summary = data_summary.dropna(subset=['Length']).astype({'Length': 'int64'})

    # Only keep desc of datasets with 1NN cost < threshold
    data_summary['1NN_cost'] = data_summary['Train '] * data_summary['Test '] * data_summary['Length']
    data_summary = data_summary[data_summary['1NN_cost'] < threshold]

    # Only consider datasets both in dir and in desc
    datasets_in_dir = os.listdir(root_dir)
    datasets_list = [d for d in datasets_in_dir if d in list(data_summary['Name'])]

    # Should be 53 for threshold = 1e7 and 80 for threshold = 1e8
    print(f"Number of datasets : {len(datasets_list)}")

    return datasets_list