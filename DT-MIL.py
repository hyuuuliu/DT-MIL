import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torchviz import make_dot, make_dot_from_trace

import matplotlib.pyplot as plt
import glob

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class mil(nn.Module):
    def __init__(self, in_dim, rnn_hidden_size, fc_hidden_size, out_dim):
        super(mil, self).__init__()
        self.rnn_layer = nn.GRU(in_dim, rnn_hidden_size, 1).cuda()
        self.fc = nn.Sequential(nn.Linear(rnn_hidden_size,fc_hidden_size),nn.Tanh()).cuda()
        
        self.logitic = nn.Sequential(nn.Linear(fc_hidden_size,out_dim),nn.Sigmoid()).cuda()
        
        
    def forward(self, inputs):
        seq_len = len(inputs)
        
        batch_size = len(inputs[0])
        out, hidden = self.rnn_layer(inputs.view(seq_len, batch_size, -1))
        in_fc = torch.transpose(out,0,1)
        out_fc = self.fc(in_fc)
        out_p = self.logitic(out_fc)
        p_max = torch.max(out_p,1)[0].cuda()        
        
        return p_max
    
    def evaluate(self, inputs):
        seq_len = len(inputs)
        
        batch_size = len(inputs[0])
        out, hidden = self.rnn_layer(inputs.view(seq_len, batch_size, -1))
        in_fc = torch.transpose(out,0,1)
        out_fc = self.fc(in_fc)
        out_p = self.logitic(out_fc)
        p_max = torch.max(out_p,1)[0].cuda()
        
        return out_p

    


in_dim = 1
out_dim = 1
rnn_hidden_size = 20 # rnn hidden size
fc_hidden_size = 500


model = mil(in_dim, rnn_hidden_size, fc_hidden_size, out_dim).cuda()



##############
# Test
##############

# test data

a = [i for i in range(1,101)]
b = []
for i in range(101):
    current = a[:i]
    b.append(np.pad(current,(0,100-len(current)),'constant',constant_values=(0)).tolist())

b = np.transpose(b).tolist()
c = [0 for i in range(0,51)] + [1 for i in range(51,101)]
inputs = Variable(torch.FloatTensor(b),requires_grad = True).cuda()
outputs = Variable(torch.FloatTensor(c),requires_grad = False).cuda()


#loss function
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002,weight_decay=0.01)

for epoch in range(10000):
    model.train()  
    optimizer.zero_grad()
    
    y_pred = model(inputs)
    y_real = outputs
    loss = criterion(y_pred, y_real)
    loss.backward()
    optimizer.step()
    if epoch%100==0:
        print('loss=',loss)