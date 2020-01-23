import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from models_torch import *
from utils import *
import argparse

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

def loss_function(pred, target):
    loss = 1/np.log(2) * F.nll_loss(pred, target)
    return loss

def train(epoch, reps=20):
    model.train()
    train_loss = 0
    for batch_idx, sample in enumerate(train_loader):
        # data = torch.from_numpy(data)
        data, target = sample['x'].to(device), sample['y'].to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = loss_function(pred, target)
        loss.backward()
        train_loss += loss.item()
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        if batch_idx % 10 == 0:
            print('====> Epoch: {} Batch {}/{} Average loss: {:.10f}'.format(
            epoch, batch_idx+1, len(train_loader), train_loss / (batch_idx+1)), end='\r', flush=True)

    print('====> Epoch: {} Average loss: {:.10f}'.format(
        epoch, train_loss / (batch_idx+1)), flush=True)

def get_argument_parser():
    parser = argparse.ArgumentParser();
    parser.add_argument('--file_name', type=str, default='xor10_small',
                        help='The name of the input file')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Name for the log file')
    parser.add_argument('--epochs', type=int, default='10',
                        help='Num of epochs')
    return parser

parser = get_argument_parser()
FLAGS = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu
num_epochs=FLAGS.epochs

batch_size=2048
timesteps=64
use_cuda = True

use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Using", device)
sequence = np.load(FLAGS.file_name + ".npy")
vocab_size = len(np.unique(sequence))
sequence = sequence

X, Y = generate_single_output_data(sequence, batch_size, timesteps)
X = X.astype('int32')
Y = Y.astype('int32')

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_dataset = CustomDL(X, Y)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True, **kwargs)

dic = {'vocab_size': vocab_size, 'emb_size': 8,
        'length': timesteps, 'jump': 16,
        'hdim1': 8, 'hdim2': 16, 'n_layers': 2,
        'bidirectional': True}

print("Vocab Size {}".format(vocab_size))
if vocab_size >= 1 and vocab_size <=3:
    dic['hdim1'] = 8
    dic['hdim2'] = 16
  
if vocab_size >= 4 and vocab_size <=8:
    dic['hdim1'] = 32
    dic['hdim2'] = 16

if vocab_size >= 10 and vocab_size < 128:
    dic['hdim1'] = 128
    dic['hdim2'] = 128
    dic['emb_size'] = 16

if vocab_size >= 128:
    dic['hdim1'] = 128
    dic['hdim2'] = 256
    dic['emb_size'] = 16

model = BootstrapNN(**dic).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(FLAGS.epochs):
    train(epoch+1)

torch.save(model.state_dict(), FLAGS.file_name + "_bstrap")

print("Done")




