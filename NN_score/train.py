import torch
import torch.nn as nn
import torch.optim as optim
from dockschnet import dockschnet


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

# Función para calcular el producto punto
def dot_product(net1_output, net2_output):
    return torch.dot(net1_output, net2_output)

# Ciclo de entrenamiento
for epoch in range(num_epochs):
    for data in dataloader:
        # Obtén tus entradas, por ejemplo, data1 y data2

        # Pasa las entradas por las redes
        output1 = net1(data1)
        output2 = net2(data2)

        # Calcula el producto punto
        dot_prod = dot_product(output1, output2)

        # Calcula la pérdida
        loss = criterion(dot_prod, target) # 'target' depende de tu tarea específica

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

