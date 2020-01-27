import numpy as np
import os
import sys
import json
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from utils import *
import tempfile
import argparse
import arithmeticcoding_fast
import struct

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"


def loss_function(pred, target):
    loss = 1/np.log(2) * F.nll_loss(pred, target)
    return loss


def decompress(model, len_series, bs, vocab_size, timesteps, device, final_step=False):
    
    if not final_step:
        num_iters = len_series // bs
        series_2d = np.zeros((bs,num_iters), dtype = np.uint8).astype('int')
        ind = np.array(range(bs))*num_iters

        f = [open(FLAGS.temp_file_prefix+'.'+str(i),'rb') for i in range(bs)]
        bitin = [arithmeticcoding_fast.BitInputStream(f[i]) for i in range(bs)]
        dec = [arithmeticcoding_fast.ArithmeticDecoder(32, bitin[i]) for i in range(bs)]

        prob = np.ones(vocab_size)/vocab_size
        cumul = np.zeros(vocab_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)

        for i in range(bs):
            for j in range(min(timesteps, num_iters)):
                series_2d[i,j] = dec[i].read(cumul, vocab_size)

        cumul = np.zeros((bs, vocab_size+1), dtype = np.uint64)

        for j in (range(num_iters - timesteps)):
            # Write Code for probability extraction
            bx = Variable(torch.from_numpy(series_2d[:,j:j+timesteps])).to(device)
            with torch.no_grad():
                model.eval()
                prob = torch.exp(model(bx)).detach().cpu().numpy()
            cumul[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)
            for i in range(bs):
                series_2d[i,j+timesteps] = dec[i].read(cumul[i,:], vocab_size)
            by = Variable(torch.from_numpy(series_2d[:, j+timesteps])).to(device)


        # close files
        for i in range(bs):
            bitin[i].close()
            f[i].close()
        return series_2d.reshape(-1)
    
    else:
        series = np.zeros(len_series, dtype = np.uint8).astype('int')
        f = open(FLAGS.temp_file_prefix+'.last','rb')
        bitin = arithmeticcoding_fast.BitInputStream(f)
        dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
        prob = np.ones(vocab_size)/vocab_size
        cumul = np.zeros(vocab_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)        

        for j in range(min(timesteps,len_series)):
            series[j] = dec.read(cumul, vocab_size)
        for i in range(len_series-timesteps):
            bx = Variable(torch.from_numpy(series[i:i+timesteps].reshape(1,-1))).to(device)
            with torch.no_grad():
                model.eval()
                prob = torch.exp(model(bx)).detach().cpu().numpy()
            cumul[1:] = np.cumsum(prob*10000000 + 1)
            series[i+timesteps] = dec.read(cumul, vocab_size)
        bitin.close()
        f.close()
        return series





class BootstrapNN(nn.Module):
    def __init__(self, vocab_size, emb_size, length, jump, hdim1, hdim2, n_layers, bidirectional):
        super(BootstrapNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.vocab_size = vocab_size
        self.len = length
        self.hdim1 = hdim1
        self.hdim2 = hdim2
        self.n_layers = n_layers
        self.jump = jump
        self.rnn_cell = nn.GRU(emb_size, hdim1, n_layers, batch_first=True, bidirectional=bidirectional)
        
        if bidirectional:
            self.lin1 = nn.Sequential(
            nn.Linear(2*hdim1*(length//jump), hdim2),
            nn.ReLU(inplace=True)
            )
            self.flin1 = nn.Linear(2*hdim1*(length//jump), vocab_size)
        else:
            self.lin1 = nn.Sequential(
            nn.Linear(hdim1*(length//jump), hdim2),
            nn.ReLU(inplace=True)
            )
            self.flin1 = nn.Linear(hdim1*(length//jump), vocab_size)
        self.flin2 = nn.Linear(hdim2, vocab_size)

    def forward(self, inp):
        emb = self.embedding(inp)
        output, hidden = self.rnn_cell(emb)
        slicedoutput = torch.flip(output, [2])[:,::self.jump,:]
        batch_size = slicedoutput.size()[0]
        flat = slicedoutput.contiguous().view(batch_size, -1)
        prelogits = x = self.lin1(flat)
        x = self.flin1(flat) + self.flin2(x)
        out = F.log_softmax(x, dim=1)

        return out

def get_argument_parser():
    parser = argparse.ArgumentParser();
    parser.add_argument('--file_name', type=str, default='xor10_comp',
                        help='The name of the input file')
    parser.add_argument('--output', type=str, default='xor10_small_decomp',
                        help='The name of the output file')
    parser.add_argument('--model_weights_path', type=str, default='xor10_small_bstrap',
                        help='Path to model weights')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU to use')
    return parser


def var_int_decode(f):
    byte_str_len = 0
    shift = 1
    while True:
        this_byte = struct.unpack('B', f.read(1))[0]
        byte_str_len += (this_byte & 127) * shift
        if this_byte & 128 == 0:
                break
        shift <<= 7
        byte_str_len += shift
    return byte_str_len

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu
    use_cuda = True

    FLAGS.temp_dir = 'temp'
    FLAGS.temp_file_prefix = FLAGS.temp_dir + "/compressed"
    if not os.path.exists(FLAGS.temp_dir):
        os.makedirs(FLAGS.temp_dir)

    f = open(FLAGS.file_name+'.params','r')
    params = json.loads(f.read())
    f.close()

    batch_size = params['bs']
    timesteps = params['timesteps']
    len_series = params['len_series']
    id2char_dict = params['id2char_dict']
    vocab_size = len(id2char_dict)

    f = open(FLAGS.file_name+'.combined','rb')
    for i in range(batch_size):
        f_out = open(FLAGS.temp_file_prefix+'.'+str(i),'wb')
        byte_str_len = var_int_decode(f)
        byte_str = f.read(byte_str_len)
        f_out.write(byte_str)
        f_out.close()
    f_out = open(FLAGS.temp_file_prefix+'.last','wb')
    byte_str_len = var_int_decode(f)
    byte_str = f.read(byte_str_len)
    f_out.write(byte_str)
    f_out.close()
    f.close()

    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    series = np.zeros(len_series,dtype=np.uint8)

    bsdic = {'vocab_size': vocab_size, 'emb_size': 8,
        'length': timesteps, 'jump': 16,
        'hdim1': 8, 'hdim2': 16, 'n_layers': 2,
        'bidirectional': True}

    if vocab_size >= 1 and vocab_size <=3:
        bsdic['hdim1'] = 8
        bsdic['hdim2'] = 16
      
    if vocab_size >= 4 and vocab_size <=9:
        bsdic['hdim1'] = 32
        bsdic['hdim2'] = 16

    if vocab_size >= 10 and vocab_size < 128:
        bsdic['hdim1'] = 128
        bsdic['hdim2'] = 128
        bsdic['emb_size'] = 16

    if vocab_size >= 128:
        bsdic['hdim1'] = 128
        bsdic['hdim2'] = 256
        bsdic['emb_size'] = 16

    model = BootstrapNN(**bsdic).to(device)
    model.load_state_dict(torch.load(FLAGS.model_weights_path))

    l = int(len(series)/batch_size)*batch_size
    
    series[:l] = decompress(model, l, batch_size, vocab_size, timesteps, device)
    if l < len_series - timesteps:
        series[l:] = decompress(model, len_series-l, 1, vocab_size, timesteps, device, final_step = True)
    else:
        f = open(FLAGS.temp_file_prefix+'.last','rb')
        bitin = arithmeticcoding_fast.BitInputStream(f)
        dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin) 
        prob = np.ones(vocab_size)/vocab_size
        
        cumul = np.zeros(vocab_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)        
        for j in range(l, len_series):
            series[j] = dec.read(cumul, vocab_size)
        
        bitin.close() 
        f.close()
    
    np.save(FLAGS.output, series)
    a = np.load('xor10_small.npy')
    print("Decompressed file is same? {}".format(np.array_equal(a,series)))
    print("Done")
    


if __name__ == "__main__":
    parser = get_argument_parser()
    FLAGS = parser.parse_args()
    main()




