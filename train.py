import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

import sys
sys.stdout.reconfigure(encoding='utf-8')

# Carregar o arquivo intents.json corretamente
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Definir caracteres a ignorar
ignore_words = ['?', '.', '!', ',', ';', ':', '(', ')', '[', ']', '{', '}']

# Processar cada padrão do intents.json
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)  # Adicionar a tag à lista de tags

    for pattern in intent['patterns']:
        w = tokenize(pattern)  # Tokenizar a sentença
        w = [word for word in w if word not in ignore_words]  # Remover pontuação
        all_words.extend(w)  # Adicionar palavras à lista geral
        xy.append((w, tag))  # Adicionar à lista de pares (sentença, tag)

# Aplicar stemming e remover duplicatas
all_words = sorted(set([stem(w) for w in all_words]))
tags = sorted(set(tags))

print(len(xy), "padrões")
print(len(tags), "tags:", tags)
print(len(all_words), "palavras únicas tratadas:", all_words)

# Criar os dados de treino
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)  # Criar bag of words
    X_train.append(bag)
    label = tags.index(tag)  # Converter tag para índice numérico
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hiperparâmetros
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(f"Input Size: {input_size}, Output Size: {output_size}")

# Criar dataset para PyTorch
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Treinar o modelo
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Final loss: {loss.item():.4f}')

# Salvar o modelo treinado
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Treinamento concluído. Arquivo salvo em {FILE}')
