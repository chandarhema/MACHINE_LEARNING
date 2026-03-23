import pandas as pd

data = pd.read_csv("OHSU_0050147_rois_ho.1D", sep="\t", header=None)
print(data)
print(data.shape)
print(data.describe())