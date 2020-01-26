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

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

class CustomDL(Dataset):

  def __init__(self, features, labels):

    self.features = features
    self.labels = labels

  def __len__(self):
    return len(self.features)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()
    feat = self.features[idx].astype('int')
    lab = self.labels[idx].astype('int')
    sample = {'x': feat, 'y': lab}

    return sample

def evaluate(model, loader, device):
    model.eval()
    pred_list = []
    with torch.no_grad():
        for sample in loader:
            data = sample['x'].to(device)
            pred_list.append(model(data).detach().cpu().numpy())

    return np.concatenate(pred_list, axis=0)

def loss_function(pred, target):
    loss = 1/np.log(2) * F.nll_loss(pred, target)
    return loss

def train(model, loader, device):
    train_loss = 0
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    pred_list = []
    for batch_idx, sample in enumerate(loader):
        # data = torch.from_numpy(data)
        model.train()
        data, target = sample['x'].to(device), sample['y'].to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = loss_function(pred, target)
        loss.backward()
        # train_loss += loss.item()
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        with torch.no_grad():
            model.eval()
            pred_list.append(model(data).detach().cpu().numpy())


    return np.concatenate(pred_list, axis=0)

def compress(model, X, Y, bs, vocab_size, timesteps, device, final_step=False):
    
    if not final_step:
        num_iters = (len(X)+timesteps) // bs
        ind = np.array(range(bs))*num_iters

        f = [open(FLAGS.temp_file_prefix+'.'+str(i),'wb') for i in range(bs)]
        bitout = [arithmeticcoding_fast.BitOutputStream(f[i]) for i in range(bs)]
        enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i]) for i in range(bs)]

        prob = np.ones(vocab_size)/vocab_size
        cumul = np.zeros(vocab_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)

        for i in range(bs):
            for j in range(min(timesteps, num_iters)):
                enc[i].write(cumul, X[ind[i],j])

        cumul = np.zeros((bs, vocab_size+1), dtype = np.uint64)

        for j in (range(num_iters - timesteps)):
            # Write Code for probability extraction
            bx = Variable(torch.from_numpy(X[ind,:])).to(device)
            by = Variable(torch.from_numpy(Y[ind])).to(device)
            with torch.no_grad():
                model.eval()
                prob = torch.exp(model(bx)).detach().cpu().numpy()
            cumul[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)
            for i in range(bs):
                enc[i].write(cumul[i,:], Y[ind[i]])
            ind = ind + 1

        # close files
        for i in range(bs):
            enc[i].finish()
            bitout[i].close()
            f[i].close()
    
    else:
        f = open(FLAGS.temp_file_prefix+'.last','wb')
        bitout = arithmeticcoding_fast.BitOutputStream(f)
        enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
        prob = np.ones(vocab_size)/vocab_size
        cumul = np.zeros(vocab_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)        

        for j in range(timesteps):
            enc.write(cumul, X[0,j])
        for i in (range(len(X))):
            bx = Variable(torch.from_numpy(X[i:i+1,:])).to(device)
            with torch.no_grad():
                model.eval()
                prob = torch.exp(model(bx)).detach().cpu().numpy()
            cumul[1:] = np.cumsum(prob*10000000 + 1)
            enc.write(cumul, Y[i])
        enc.finish()
        bitout.close()
        f.close()
    
    return





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
    parser.add_argument('--file_name', type=str, default='xor10_small',
                        help='The name of the input file')
    parser.add_argument('--model_weights_path', type=str, default='bstrap',
                        help='Path to model weights')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU to use')
    parser.add_argument('--output', type=str, default='comp',
                        help='Name of the output file')
    parser.add_argument('--params', type=str, default='params_xor10_small',
                        help='Name of the output file')
    return parser


def var_int_encode(byte_str_len, f):
    while True:
        this_byte = byte_str_len&127
        byte_str_len >>= 7
        if byte_str_len == 0:
                f.write(struct.pack('B',this_byte))
                break
        f.write(struct.pack('B',this_byte|128))
        byte_str_len -= 1

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu

    batch_size=512
    timestep=64
    use_cuda = True

    with open(FLAGS.params, 'r') as f:
        params = json.load(f)

    FLAGS.temp_dir = 'temp'
    FLAGS.temp_file_prefix = FLAGS.temp_dir + "/compressed"
    if not os.path.exists(FLAGS.temp_dir):
        os.makedirs(FLAGS.temp_dir)

    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    sequence = np.load(FLAGS.file_name + ".npy")
    vocab_size = len(np.unique(sequence))
    sequence = sequence

    sequence = sequence.reshape(-1)
    series = sequence.copy()
    data = strided_app(series, timestep+1, 1)
    X = data[:, :-1]
    Y = data[:, -1]
    X = X.astype('int')
    Y = Y.astype('int')

    params['len_series'] = len(series)
    params['bs'] = batch_size
    params['timesteps'] = timestep

    with open(FLAGS.output+'.params','w') as f:
        json.dump(params, f, indent=4)


    model = BootstrapNN(vocab_size=vocab_size,
                        emb_size=8,
                        length=timestep,
                        jump=16,
                        hdim1=8,
                        hdim2=16,
                        n_layers=2,
                        bidirectional=True).to(device)
    model.load_state_dict(torch.load(FLAGS.file_name +"_bstrap"))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # dataset = CustomDL(X, Y_original)
    # loader = torch.utils.data.DataLoader(dataset,
    #                                     batch_size=batch_size,
    #                                     shuffle=False, **kwargs)

    # probs = evaluate(model, loader, device)
    # print(probs.shape)
    # np.save('probs_bs{}'.format(batch_size), probs)

    # probs = train(model, loader, device)
    # print(probs.shape)
    # np.save('probs_2', probs)

    l = int(len(series)/batch_size)*batch_size
    
    compress(model, X, Y, batch_size, vocab_size, timestep, device)
    if l < len(series)-timestep:
        compress(model, X[l:], Y[l:], 1, vocab_size, timestep, device, final_step = True)
    else:
        f = open(args.temp_file_prefix+'.last','wb')
        bitout = arithmeticcoding_fast.BitOutputStream(f)
        enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout) 
        prob = np.ones(vocab_size)/vocab_size
        
        cumul = np.zeros(vocab_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)        
        for j in range(l, len(series)):
                enc.write(cumul, series[j])
        enc.finish()
        bitout.close() 
        f.close()
    
    print("Done")
    
    # # combine files into one file
    # f = open(args.output_file_prefix+'.combined','wb')
    # for i in range(batch_size):
    #     f_in = open(args.temp_file_prefix+'.'+str(i),'rb')
    #     byte_str = f_in.read()
    #     byte_str_len = len(byte_str)
    #     var_int_encode(byte_str_len, f)
    #     f.write(byte_str)
    #     f_in.close()
    # f_in = open(args.temp_file_prefix+'.last','rb')
    # byte_str = f_in.read()
    # byte_str_len = len(byte_str)
    # var_int_encode(byte_str_len, f)
    # f.write(byte_str)
    # f_in.close()
    # f.close()
    # shutil.rmtree(args.temp_dir)


if __name__ == "__main__":
    parser = get_argument_parser()
    FLAGS = parser.parse_args()
    main()




