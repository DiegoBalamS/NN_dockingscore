import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from dockschnet import dockschnet
from pretrain_trfm import TrfmSeq2seq
from build_vocab import WordVocab
from utils import split
import numpy as np
from lightningmodule import MyModel
import pytorch_lightning as pl
from lightningmodule import MyDataset
from torch.utils.data import Dataset, DataLoader

def prepare(file):
    # Cargar un archivo PDB
    mol = Chem.MolFromPDBFile(file, removeHs=False)

    # Añadir hidrógenos explícitos (si es necesario)
    mol = Chem.AddHs(mol)

    # Extraer posiciones de átomos
    atoms = mol.GetAtoms()
    num_atoms = len(atoms)
    positions = [mol.GetConformer().GetAtomPosition(i) for i in range(num_atoms)]
    positions = torch.tensor(positions, dtype=torch.float)

    # Extraer tipos de átomos
    atomic_numbers = [atom.GetAtomicNum() for atom in atoms]
    atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)

    # Crear edge_index (asumiendo que cada enlace es una arista)
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append((i, j))
        edge_index.append((j, i))  # Añadir en ambas direcciones
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Crear objeto Data para torch_geometric
    data = Data(x=atomic_numbers.view(-1, 1), pos=positions, edge_index=edge_index)

    return data

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

#Training data
data = 'project/training_info.txt'


# we separate the incoming information and the outcome
input = []
targets = []

pdb_locations=['project/pdb/cluster_0_random_structure.pdb','project/pdb/cluster_1_random_structure.pdb','project/pdb/cluster_2_random_structure.pdb','project/pdb/cluster_3_random_structure.pdb','project/pdb/cluster_4_random_structure.pdb','project/pdb/cluster_5_random_structure.pdb','project/pdb/cluster_6_random_structure.pdb','project/pdb/cluster_7_random_structure.pdb','project/pdb/cluster_8_random_structure.pdb','project/pdb/cluster_9_random_structure.pdb','project/pdb/cluster_10_random_structure.pdb','project/pdb/cluster_11_random_structure.pdb','project/pdb/cluster_12_random_structure.pdb','project/pdb/cluster_13_random_structure.pdb','project/pdb/cluster_14_random_structure.pdb','project/pdb/cluster_15_random_structure.pdb']

with open(data, 'r') as file:
    for linea in file:
        
        elements = linea.strip().split()
        pdb_file=pdb_locations[int(elements[1])]
        data2=prepare(pdb_file)
        el1=data2.x.ravel()
        el2=data2.pos

        x_split=[split(elements[0])]
        xid, _ = get_array(x_split)

        input.append([xid,el1,el2])
        #input.append(elements[:2])
        targets.append(elements[2])

train_input, test_input, train_targets, test_targets = train_test_split(input, targets, test_size=0.2, random_state=42)
train_targets = np.array([float(string) for string in train_targets])
#train_targets = torch.tensor(train_targets)

model = MyModel(vocab,pad_index=0, unk_index=1, eos_index=2, sos_index=3, mask_index=4)

model.net1.load_state_dict(torch.load('project/trfm_12_23000.pkl', map_location=torch.device('cpu')))

model.net1.eval()

trainer = pl.Trainer(max_epochs=10)
