import random
import json
import torch
import sys
import os
from flask import Flask, request, jsonify
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)

CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'

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

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

@app.route('/message', methods=['POST'])
def message():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'Mensagem não identificada'}), 400

    print(f"{CYAN}Mensagem: {user_message}{RESET}")

    sentence = tokenize(user_message)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    
    probs = torch.softmax(output, dim=1)
    
    top3_prob, top3_idx = torch.topk(probs, 3)

    print(f"{CYAN}Tags mais prováveis:{RESET}")
    for i in range(3):
        tag = tags[top3_idx[0][i].item()]
        prob = top3_prob[0][i].item()
        color = GREEN if i == 0 else YELLOW
        print(f"{color}{i+1}. Tag: {tag}, Prob: {prob:.4f}{RESET}")

    high_prob_tags = [(tags[top3_idx[0][i].item()], top3_prob[0][i].item()) for i in range(3) if top3_prob[0][i].item() > 0.75]

    if len(high_prob_tags) > 1:
        main_tag = high_prob_tags[0][0]
        for intent in intents['intents']:
            if main_tag == intent["tag"]:
                response = random.choice(intent['responses'])
                print(f"{CYAN}Resposta: Vamos focar em um tema por vez.\n{response}{RESET}")
                return jsonify({'response': f"Vamos focar em um tema por vez.\n{response}"})
    
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                print(f"{CYAN}Resposta: {response}{RESET}")
                return jsonify({'response': response})
    else:
        response = "Desculpe, não entendi..."
        print(f"{CYAN}Resposta: {response}{RESET}")
        return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))