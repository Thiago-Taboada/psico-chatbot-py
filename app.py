import random
import json
import torch
import sys
from flask import Flask, request, jsonify
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

import nltk
nltk.download('punkt')
nltk.download('rslp')

sys.stdout.reconfigure(encoding='utf-8')

# Iniciar a aplicação Flask
app = Flask(__name__)

# Carregar os dados do modelo
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

@app.route('/message', methods=['POST'])
def message():
    user_message = request.json.get('message')  # A mensagem do usuário
    if not user_message:
        return jsonify({'error': 'Mensagem não fornecida'}), 400

    sentence = tokenize(user_message)
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
                return jsonify({'response': random.choice(intent['responses'])})
    else:
        return jsonify({'response': "Desculpe, não entendi..."})

if __name__ == '__main__':
    # Inicia o servidor Flask
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))
