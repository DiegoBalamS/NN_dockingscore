class SMILESModel(torch.nn.Module):
    def __init__(self):
        super(SMILESModel, self).__init__()
        # Inicializar el modelo SMILES-Transformer aquí

    def forward(self, smiles_string):
        # Procesar la cadena SMILES y devolver el vector
        return vector_from_smiles(smiles_string)
