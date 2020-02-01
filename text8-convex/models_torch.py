import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np

class BootstrapNN(nn.Module):
    def __init__(self, vocab_size, emb_size, length, jump, hdim1, hdim2, n_layers, bidirectional):
        super(BootstrapNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.vocab_size = vocab_size
        self.len = length
        self.hdim1 = hdim1
        self.hdim2 = hdim2
        self.n_layers = n_layers
        self.bidirectional = bidirectional
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
        slicedoutput = torch.flip(output, [1])[:,::self.jump,:]
        batch_size = slicedoutput.size()[0]
        flat = slicedoutput.contiguous().view(batch_size, -1)
        prelogits = x = self.lin1(flat)
        x = self.flin1(flat) + self.flin2(x)
        out = F.log_softmax(x, dim=1)

        return out

class CombinedNN(nn.Module):
    def __init__(self, bsNN, vocab_size, emb_size, length, hdim):
        super(CombinedNN, self).__init__()
        self.bsembedding = bsNN.embedding
        self.bsrnn_cell = bsNN.rnn_cell
        self.bslin1 = bsNN.lin1
        self.bsjump = bsNN.jump
        if bsNN.bidirectional:
            # self.flin1 = nn.Linear(2*bsNN.hdim1*(length//bsNN.jump), vocab_size)
            self.flat2_size = 2*bsNN.hdim1*(length//bsNN.jump) + emb_size*length
            self.bsflin1 = bsNN.flin1
        else:
            # self.flin1 = nn.Linear(bsNN.hdim1*(length//bsNN.jump), vocab_size)
            self.flat2_size = bsNN.hdim1*(length//bsNN.jump) + emb_size*length
            self.bsflin1 = bsNN.flin1


        # self.flin2 = nn.Linear(bsNN.hdim2, vocab_size)
        self.bsflin2 = bsNN.flin2

        self.hdim = hdim
        self.embedding = nn.Embedding(vocab_size, emb_size)

        self.layer11 = nn.Sequential(
            nn.Linear(self.flat2_size, hdim),
            nn.ReLU(inplace=False)
            )

        self.layer12 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(hdim, hdim),
            nn.ReLU(inplace=False),
            nn.Linear(hdim, hdim)
            )

        self.layer13 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(hdim, hdim),
            nn.ReLU(inplace=False),
            nn.Linear(hdim, hdim)
            )

        self.layer2 = nn.Sequential(
            nn.Linear(self.flat2_size, hdim),
            nn.ReLU(inplace=False),
            nn.Linear(hdim, hdim),
            nn.ReLU(inplace=False)
            )

        self.last_lin1 = nn.Linear(self.flat2_size, vocab_size)
        self.last_lin2 = nn.Linear(hdim, vocab_size)
        self.last_lin3 = nn.Linear(hdim, vocab_size)

        self.weight = nn.Parameter(torch.zeros([1], dtype=torch.float32), requires_grad=True)

        self.final = nn.Linear(vocab_size*3, vocab_size)

    def forward(self, inp):
        emb = self.bsembedding(inp)
        output, hidden = self.bsrnn_cell(emb)
        slicedoutput = torch.flip(output, [1])[:,::self.bsjump,:]
        batch_size = slicedoutput.size()[0]
        flat = slicedoutput.contiguous().view(batch_size, -1)
        prelogits = x = self.bslin1(flat)
        new_logits = self.bsflin1(flat) + self.bsflin2(x)
        out = F.log_softmax(new_logits, dim=1)

        emb = self.embedding(inp)
        d = emb.view(batch_size, -1)
        flat2 = torch.cat((d, flat), 1)

        d = self.layer11(flat2)
        x = d
        d = self.layer12(d) + x
        x = d
        d = self.layer13(d) + x

        e = self.layer2(flat2)

        next_layer = torch.cat((self.last_lin1(flat2), self.last_lin2(d), self.last_lin3(e)), 1)
        # print(self.weight, 1- self.weight, self.final(next_layer))
        pred = self.final(next_layer)
        final_logits = torch.sigmoid(self.weight) * pred + (1 - torch.sigmoid(self.weight)) * new_logits
        out = F.log_softmax(final_logits, dim=1)
        return out, F.log_softmax(pred, dim=1)