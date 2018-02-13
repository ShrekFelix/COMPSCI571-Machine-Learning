import numpy as np

def load_CSV(file_name):
    tmp = np.loadtxt(file_name, dtype=np.str, delimiter=",")
    features = tmp[1:,:-1].astype(np.float)# load features
    labels = tmp[1:,-1].astype(np.float)# load labels
    return features, labels # return ndarray
