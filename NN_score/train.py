model = DockingScorePredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for pdb_data, smiles_string, true_score in data_loader:
        optimizer.zero_grad()
        predicted_score = model(pdb_data, smiles_string)
        loss = loss_function(predicted_score, true_score)
        loss.backward()
        optimizer.step()
    # Validar el modelo regularmente
