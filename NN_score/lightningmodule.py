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

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, pdb_locations, vocab, transform=None):
        self.inputs = inputs
        self.targets = targets
        self.pdb_locations = pdb_locations
        self.vocab = vocab
        self.transform = transform
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input = self.inputs[idx]
        target = self.targets[idx]
        # Transform input as needed, e.g., tokenize SMILES, load PDB
        # Placeholder for transformation logic
        return input, target

class MyModel(pl.LightningModule):
    def __init__(self, vocab_size, pad_index, unk_index, eos_index, sos_index, mask_index):
        super().__init__()
        self.net1 = TrfmSeq2seq(vocab_size, 256, vocab_size, 4)
        self.net2 = dockschnet(hidden_channels=16, num_filters=16, num_interactions=3)
        self.pad_index = pad_index
        self.unk_index = unk_index
        self.eos_index = eos_index
        self.sos_index = sos_index
        self.mask_index = mask_index
        self.vocab = WordVocab.load_vocab('project/vocab.pkl')

    def forward(self, x):
        # Procesar SMILES a través de net1
      # Asumimos que ya has convertido los SMILES en la representación adecuada (ID de tokens, máscaras, etc.)
      smiles_features = self.net1(smiles)  # Asegúrate de que esto devuelva el vector de características deseado
    
    # Procesar datos PDB a través de net2
    # Asumimos que pdb_data ya está en la forma adecuada que net2 espera (e.g., tensores de posiciones, tipos atómicos, etc.)
      pdb_features = self.net2(pdb_data)  # Asegúrate de que esto devuelva el vector de características deseado
    
    # Calcular el producto punto entre los vectores de características
    # Asegúrate de que ambos vectores sean de la misma dimensión
      dot_product = torch.dot(smiles_features, pdb_features)
    
      return dot_product
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        # Logic for a single training step
        # Calculate loss here
        loss = torch.tensor(0) # Placeholder
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        # Logic for a single validation step
        # Calculate and log validation loss/metrics
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def prepare_data(self):
        # Prepare or load your data here
        pass

    def train_dataloader(self):
        # Return DataLoader for training
        dataset = MyDataset(...) # Fill with your actual data and preprocessing
        return DataLoader(dataset, batch_size=32, shuffle=True)

    def val_dataloader(self):
        # Return DataLoader for validation
        dataset = MyDataset(...) # Fill with your actual data and preprocessing
        return DataLoader(dataset, batch_size=32)

# Use your actual vocab size and indices
model = MyModel(vocab_size=len(vocab), pad_index=0, unk_index=1, eos_index=2, sos_index=3, mask_index=4)

# Load model weights
model.net1.load_state_dict(torch.load('project/trfm_12_23000.pkl', map_location=torch.device('cpu')))
model.net1.eval()

trainer = pl.Trainer(max_epochs=10)
trainer.fit(model)

