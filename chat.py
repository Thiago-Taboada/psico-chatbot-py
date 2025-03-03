import random
import json
import torch
import sys

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Garantir que a codificação seja UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Verificar se o dispositivo disponível é CUDA (GPU) ou CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carregar o arquivo de intents
with open('intents.json', 'r', encoding='utf-8') as json_data:  # Especificando a codificação UTF-8
    intents = json.load(json_data)

# Carregar o modelo treinado
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Inicializar o modelo
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "IA Belle"
print("Conversa iniciada! (digite 'sair' para encerrar)")

while True:
    sentence = input("Você: ")
    if sentence.lower() == "sair":
        break

    # Tokenização da sentença de entrada
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Fazer a previsão
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    # Obter a probabilidade da previsão
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Exibir a resposta
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: Desculpe, não entendi...")
