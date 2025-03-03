import random
import json
import torch
import sys
import argparse

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

sys.stdout.reconfigure(encoding='utf-8')

parser = argparse.ArgumentParser()
parser.add_argument('--message', type=str, required=True, help='Mensagem que será enviada ao bot')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

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

# Processar a mensagem e obter a resposta do bot
sentence = args.message  # A mensagem do usuário
sentence = tokenize(sentence)
X = bag_of_words(sentence, all_words)
X = X.reshape(1, X.shape[0])
X = torch.from_numpy(X).to(device)

output = model(X)
_, predicted = torch.max(output, dim=1)

tag = tags[predicted.item()]

probs = torch.softmax(output, dim=1)
prob = probs[0][predicted.item()]
if prob.item() > 0.75:
    for intent in intents['intents']:
        if tag == intent["tag"]:
            print(random.choice(intent['responses']))
else:
    print("Desculpe, não entendi...")