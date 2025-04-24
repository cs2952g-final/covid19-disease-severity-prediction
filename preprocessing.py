# function to get counts per covid severity group
#reads lower count --> samples other groups based on smallest severity group. 

# imports
import pandas as pd 
import sys 
import numpy as np
import anndata as ad
#import pooch
import scanpy as sc
import random

# get filters
from filters import filter_data_freq, filter_highly_variable_genes, minimize_batch_effects, normalize_and_log

# set up scanpy
#sc.settings.set_figure_params(dpi=50, facecolor="white")

# bring in data 
# combat_file = '/Users/skylarwalters/Desktop/e243d7cd-9693-4396-8bb7-d5716782076b/COMBAT2022.h5ad'
# print(f'Reading file: {combat_file}')
# adata = sc.read_h5ad(combat_file)
# print('File read!')
# print(f'Shape: {adata.shape}\n')

def filter_data(raw_data, min_counts, n_top_genes):
    '''
    Apply filters from filters.py to the data to filter for quality, variable, etc.
    '''
    filtered = filter_data_freq(raw_data, min_counts)
    filtered = filter_highly_variable_genes(ad.AnnData(filtered), n_top_genes)

    # filter attributes of data
    filtered = filtered[(filtered.obs.disease != 'influenza') & (filtered.obs.Smoking == 'never or unknown') & (~filtered.obs.Source.isin(["Flu", "Sepsis", "COVID_LDN", "COVID_HCW_MILD"]))]
    print("Verification of filtered keys:")
    print(filtered.obs_keys())
    
    print('Data filtering complete')
    return filtered
    #return normalize_and_log(filtered)

def severity_cts(adata):
    '''
    Return tuple of (severity, ct) with the lowest amount.
    '''
    severity_to_qty = {severity : 0 for severity in ['COVID_CRIT', 'COVID_MILD', 'COVID_SEV', 'HV']}
    return_sev, return_ct = 'Error', float('inf')
    
    for elem in severity_to_qty.keys():
        severity_to_qty[elem] = len(adata[adata.obs['Source'].isin([elem])])
        if severity_to_qty[elem] < return_ct:
            return_sev = elem
            return_ct = severity_to_qty[elem]

    print(f'Minimum severity count found: {return_sev}, {return_ct}')
    return return_sev, return_ct

def get_cell_data(adata, cell_types=['B cell', 'natural killer cell', 'dendritic cell', 'CD4-positive, alpha-beta T cell', 'classical monocyte', 'double-positive, alpha-beta thymocyte', 'hematopoietic stem cell']):
    '''
    Creates cell_data, a dict mapping to adata objects for its corresponding
    cell type. 
    '''
    cell_data = {}

    cell_training_data = {}
    cell_testing_data = {}

    # iterate through cell types
    for cell_type in cell_types:
        print(f'Processing {cell_type}')
        # 1. create data object for the cell type
        cell_data[cell_type] = adata[(adata.obs.cell_type == cell_type)]
        # 2. get severity and number of samples from new adata subset
        _, n_samples = severity_cts(cell_data[cell_type])
        
        # 3. subsample based on minimum count
        balanced = None
        for severity in ['COVID_CRIT', 'COVID_MILD', 'COVID_SEV', 'HV']:
            # get subset of length n_samples that contains the given severity
            mask = cell_data[cell_type].obs.Source.isin([severity])
            subset = cell_data[cell_type][mask]
            
            # add subset to balanced list
            if balanced is None:
                balanced = subset[:n_samples] if len(subset) > n_samples else subset
            else:
                new = subset[:n_samples] if len(subset) > n_samples else subset
                balanced = ad.concat([balanced, new])
            print(f'Balanced anndata check: {type(balanced)}')
        
        # split test and train for balanced data
        cutoff = int(len(balanced) * 0.8)

        # get indices from balanced to use. 
        indices = np.random.permutation(balanced.n_obs)

        train_indices = indices[:cutoff]
        test_indices = indices[cutoff:]

        # susbset
        train_obs_names = balanced.obs_names[train_indices]
        test_obs_names = balanced.obs_names[test_indices]

        # add to training and testing subsets
        cell_training_data[cell_type] = balanced[train_obs_names].copy()
        cell_testing_data[cell_type] = balanced[test_obs_names].copy()


        #cell_training_data[cell_type] = balanced.chunk_X(select=cutoff, replace=False)
        #cell_training_data[cell_type] = random.sample(balanced, cutoff)
        #cell_testing_data[cell_type] = ad.concat([balanced, cell_training_data[cell_type]]).drop_duplicates(keep=False)

        # CHECK CELL TESTING DATA.
        print(f'Verfication: \nTraining type:{type(cell_training_data[cell_type])} \nTesting type:{type(cell_testing_data[cell_type])}')

        print(f'{cell_type} balanced. \nTraining shape: {cell_training_data[cell_type].X.shape} \nTesting shape: {cell_testing_data[cell_type].X.shape}')
    return cell_training_data, cell_testing_data

# run!
# print("Starting filter process:")
# filtered_data = filter_data(adata, 6, 10)

# print("Retrieving cell data")
# testing, training,  = get_cell_data(filtered_data)
# # to access cell matrix: cell_data[cell type].X


def get_mats(cell_data):
    '''
    returns all cell x gene matrices 
    '''
    cell_by_gene = {}
    for cell_type in cell_data:
        cell_by_gene.update({cell_type : cell_data[cell_type].X})
    
    return cell_by_gene

def get_data(combat_file, min_counts, n_top_genes):
    print(f'Reading file: {combat_file}')
    adata = sc.read_h5ad(combat_file)
    print('File read!')
    print(f'Shape: {adata.shape}\n')

    print("Starting filter process:")
    filtered_data = filter_data(adata, 6, 1000)

    print("Retrieving cell data")
    testing, training,  = get_cell_data(filtered_data)

    for cell_type in training:
        training[cell_type].write(f'training/{cell_type}_training')
        testing[cell_type].write(f'testing/{cell_type}_testing')
    
    return testing, training