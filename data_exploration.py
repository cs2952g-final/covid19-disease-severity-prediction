# file to preprocess Seurat object from COMBAT paper 

import pandas as pd 
import sys 
import numpy as np
import anndata as ad
import pooch
import scanpy as sc

sc.settings.set_figure_params(dpi=50, facecolor="white")

# bring in data 
combat_file = 'data/COMBAT_all_20250411.h5ad'
adata = sc.read_h5ad(combat_file)

### full cell x gene matrix 
# expression_data = adata.X

### names of cells and genes
# cell_ids = adata.obs_names
# gene_ids = adata.var_names

### larger cell type categories
# categories = adata.obs["GEX_region"].unique())

### get per-gene qc metrics (good for filtering later)
# sc.pp.calculate_qc_metrics(adata, expr_type = 'counts', var_type = 'genes')

### get per-cell-type patient case counts for balancing purposes

# filter out anything we don't want
cell_data = adata.obs[(adata.obs['disease'] != 'influenza') & (adata.obs['Smoking'] == 'never or unknown') & (~adata.obs['Source'].isin(["Flu", "Sepsis", "COVID_LDN", "COVID_HCW_MILD"]))]

# b cells
b_cell_data = cell_data[(cell_data['cell_type'] == 'B cell')]
print("b cell counts:")
print(b_cell_data['Source'].value_counts())
print("\n")

# NKC
nkc_data = cell_data[cell_data['cell_type'] == 'natural killer cell']
print("nkc counts:")
print(nkc_data['Source'].value_counts())
print("\n")

# dendritic
dendritic_data = cell_data[cell_data['cell_type'] == 'dendritic cell']
print("dendritic counts:")
print(dendritic_data['Source'].value_counts())
print("\n")

# t cells 
t_cell_data = cell_data[cell_data['cell_type'] == 'CD4-positive, alpha-beta T cell']
print("T-cell counts:")
print(t_cell_data['Source'].value_counts())
print("\n")

# monocytes
monocyte_data = cell_data[cell_data['cell_type'] == 'classical monocyte']
print("monocyte counts:")
print(monocyte_data['Source'].value_counts())
print("\n")

# thymocytes
thymocyte_data = cell_data[cell_data['cell_type'] == 'double-positive, alpha-beta thymocyte']
print("thymocyte counts:")
print(thymocyte_data['Source'].value_counts())
print("\n")

# stem cells
stem_cell_data = cell_data[cell_data['cell_type'] == 'hematopoietic stem cell']
print("stem cell counts:")
print(stem_cell_data['Source'].value_counts())
print("\n")

#### remove low-quality data from each ####
