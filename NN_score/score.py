class DockingScorePredictor(torch.nn.Module):
    def __init__(self):
        super(DockingScorePredictor, self).__init__()
        self.pdb_model = PDBModel()
        self.smiles_model = SMILESModel()

    def forward(self, pdb_data, smiles_string):
        pdb_vector = self.pdb_model(pdb_data)
        smiles_vector = self.smiles_model(smiles_string)
        # Calcular el producto punto para la puntuación
        docking_score = torch.dot(pdb_vector, smiles_vector)
        return docking_score
