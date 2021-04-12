# SSNP

This is the repository containing the source code for the paper titled "Self-supervised Dimensionality Reduction with Neural Networks and Pseudo-labeling", by M. Espadoto, N. S. T. Hirata and A. C. Telea, presented at IVAPP 2021.

## Python Setup

- Install Anaconda Python
- Install tensorflow, MulticoreTSNE and UMAP

```
$ pip install tensorflow-gpu multicoretsne umap-learn
```

## Obtain Datasets and Run Experiments
```
$ cd code
$ python get_data.py
$ python experiment1_direct.py
$ python experiment2_inverse.py
$ python experiment3_clustering.py
```
