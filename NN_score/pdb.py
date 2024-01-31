import pandas as pd
import torch
from rdkit import Chem
from rdkit import AllChem
from torch_geometric.data import Data

def prepare(PBD_file):
    # Cargar un archivo PDB
    mol = Chem.MolFromPDBFile(PDB_file, removeHs=False)

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
