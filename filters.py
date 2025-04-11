import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
'''
Here, we are working with a number of different filters
to refine our data. These matrices are fairly sparse,
so we want to subset by:

1) Interested cell types
2) Data quality (ie., remove low-resolution genes, low-expresion genes in >3 cells)
3) gene variability.

Highly-variable genes are desirable as they show signs of 
selection. Therefore, we can create a ranking function to 
rank genes by varability and potential search interest.
'''

def filter_data_quality(annDataObj):
    metrics = sc.pp.calculate_qc_metrics(
        adata=annDataObj
    )

def filter_data_freq(annDataObj, min_genes, min_counts):
    '''
    returns:
        cells_subset, a bool mask ndarray for filters
        number_per_cell, an nd array storing the n counts per gene.
    '''
    return sc.pp.filter_cells(annDataObj, min_genes, min_counts)

def filter_highly_variable_genes(annDataObj, n_top_genes):
    '''
    Selects and subsets to the 'n_top_genes' number of highly variable genes.
    '''
    sc.pp.highly_variable_genes(annDataObj, n_top_genes=n_top_genes, subset=True)
    return annDataObj

def minimize_batch_effects(annDataObj):
   '''
   Outputs a batch-corrected np.array of the corrected data matrix
   '''
   return sc.pp.combat(annDataObj)

def normalize_and_log(annDataObj, target_sum=1e4):
    '''
    Normalizes total counts to target_sum and applies log1p transform.
    '''
    # there's also an option to exclude very highly expressed genes if we want (unchecked for now)
    sc.pp.normalize_total(annDataObj, target_sum=target_sum)
    sc.pp.log1p(annDataObj)
    return annDataObj