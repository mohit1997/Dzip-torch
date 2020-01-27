import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class CustomDL(Dataset):

  def __init__(self, features, labels):

    self.features = features
    self.labels = labels

  def __len__(self):
    return len(self.features)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()
    feat = self.features[idx]
    lab = self.labels[idx]
    sample = {'x': feat, 'y': lab}

    return sample

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n), writeable=False)

def generate_single_output_data(series,batch_size,time_steps):
    series = series.reshape(-1)
    series = series.copy()
    data = strided_app(series, time_steps+1, 1)
    l = int(len(data)/batch_size) * batch_size

    data = data[:l] 
    X = data[:, :-1]
    Y = data[:, -1]

    return X,Y
