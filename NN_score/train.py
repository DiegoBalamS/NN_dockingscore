import torch
import torch.nn as nn
import torch.optim as optim
from dockschnet import dockschnet
from pretrain_trfm import TrfmSeq2seq
from build_vocab import WordVocab
from utils import split
from sklearn.model_selection import train_test_split

import numpy as np

#Training data
data = 'project/training_info.txt'

# we separate the incoming information and the outcome
input = []
output = []

with open(data, 'r') as file:
    for linea in file:
        
        elements = linea.strip().split()


        input.append(elements[:2])
        output.append(elements[2])


array_vectores = np.array(lista_vectores)
array_elementos = np.array(lista_elementos)




def get_inputs(sm):
    seq_len = 220
    sm = sm.split()
    if len(sm)>218:
        print('SMILES is too long ({:d})'.format(len(sm)))
        sm = sm[:109]+sm[-109:]
    ids = [vocab.stoi.get(token, unk_index) for token in sm]
    ids = [sos_index] + ids + [eos_index]
    seg = [1]*len(ids)
    padding = [pad_index]*(seq_len - len(ids))
    ids.extend(padding), seg.extend(padding)
    return ids, seg

def get_array(smiles):
    x_id, x_seg = [], []
    for sm in smiles:
        a,b = get_inputs(sm)
        x_id.append(a)
        x_seg.append(b)
    return torch.tensor(x_id), torch.tensor(x_seg)

pad_index = 0    
unk_index = 1
eos_index = 2
sos_index = 3
mask_index = 4

vocab = WordVocab.load_vocab('project/vocab.pkl')

net1 = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
net1.load_state_dict(torch.load('project/trfm_12_23000.pkl',map_location=torch.device('cpu')))
net1.eval()

net2=dockschnet(hidden_channels=16, num_filters=16, num_interactions=3)

# Definición del optimizador
optimizer = optim.Adam(list(net1.parameters()) + list(net2.parameters()), lr=0.001)

#Split the training data and the test data
train_vect, test_vect, train_out, test_out = train_test_split(input, output, test_size=0.2, random_state=42)

# function to get the dot product
def dot_product(net1_output, net2_output):
    return torch.dot(net1_output, net2_output)

# Training
for epoch in range(num_epochs):
    for dat in train_vect:
        #Get data1 and data2 
        x_split=[split(dat[0])]
        xid, _ = get_array(x_split)
        data1=trfm.encode(torch.t(xid))

        pdb_file=
        data2=prepare(pdb_file)
        
        
        output1 = net1(data1)
        output2 = net2(data2.x.ravel(),data2.pos)

        
        dot_prod = dot_product(output1, output2)

        # Loss function
        loss = criterion(dot_prod, target) # 'target' depende de tu tarea específica

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

