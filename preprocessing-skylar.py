
# function to get counts per covid severity group
#reads lower count --> samples other groups based on smallest severity group. 

# imports
import pandas as pd 
import sys 
import numpy as np
import anndata as ad
import pooch
import scanpy as sc

# get filters
from filters import filter_data_quality, filter_data_freq, filter_highly_variable_genes, minimize_batch_effects, normalize_and_log

# set up scanpy
sc.settings.set_figure_params(dpi=50, facecolor="white")

# bring in data 
combat_file = '/Users/hsnell/Downloads/COMBAT_all_20250411.h5ad'
adata = sc.read_h5ad(combat_file)

def filter_data(raw_data, min_genes, min_counts, n_top_genes):
    '''
    Apply filters from filters.py to the data to filter for quality, variable, etc.
    '''
    filtered = filter_data_quality(raw_data)
    filtered = filter_data_freq(filtered, min_genes, min_counts)
    filtered = filter_highly_variable_genes(filtered, n_top_genes)
    filtered = minimize_batch_effects(filtered)

    # filter attributes of data
    filtered = filtered.obs[(adata.obs['disease'] != 'influenza') & (adata.obs['Smoking'] == 'never or unknown') & (~adata.obs['Source'].isin(["Flu", "Sepsis", "COVID_LDN", "COVID_HCW_MILD"]))]
    return normalize_and_log(filtered)

def get_cell_data(adata, cell_types=['B cell', 'natural killer cell', 'dendritic cell', 'CD4-positive, alpha-beta T cell', 'classical monocyte', 'double-positive, alpha-beta thymocyte', 'hematopoietic stem cell']):
    '''
    return a dict mapping from cell type to the cell type's anndata file
    '''
    cell_data = {}
    for cell_type in cell_types:
        cell_data[cell_type] = adata[(adata['cell_type'] == cell_type)]
    
    return cell_data

def severity_cts(adata):
    '''
    Return tuple of (severity, ct) with the lowest amount.
    '''
    severity_to_qty = {severity : 0 for severity in ['COVID_CRIT', 'COVID_MILD', 'COVID_SEV', 'HV']}
    return_sev, return_ct = 'Error', float('inf')
    
    for elem in severity_to_qty.keys():
        severity_to_qty[elem] = len(adata[adata.obs['Source'].isin(elem)])
        if severity_to_qty[elem] < return_ct:
            return_sev = elem
            return_ct = severity_to_qty[elem]

    return return_sev, return_ct
    
# generic inputs to preproc: cell type(s), f/e severity --> 
# get covid severe data 
#then concat...?