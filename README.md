<!-- ## Assista ao Tutorial
[![Texto alternativo](https://img.youtube.com/vi/RpWeNzfSUHw/hqdefault.jpg)](https://www.youtube.com/watch?v=RpWeNzfSUHw&list=PLqnslRFeH2UrFW4AUgn-eY37qOAWQpJyg) -->

# Instalação

## Criar um ambiente

Escolha a opção que preferir (por exemplo, `conda` ou `venv`)

```console
mkdir psico-chatbot
$ cd psico-chatbot
$ python -m venv venv
```

## Ativar o ambiente

No Windows:

```console
venv\Scripts\activate
```

### Instalar Flask, PyTorch, Nltk e dependências

Execute:

```console
pip install flask torch nltk numpy
```

Se encontrar um erro na primeira execução, também será necessário instalar as dependências do nltk:

```console
python nltk_utils.py
```

OU

Execute isso uma vez no seu terminal:

```console
$ python
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('rslp')
```

## Uso

Primeiro, treine o modelo com:

```console
python train.py
```

Isso criará o arquivo `data.pth`. Após isso, execute:

```console
python chat.py
```

Isso executará o chatbot no terminal e ficará aguardando as interações.

### Ativar API HTTP com Flask

Para ativar a API HTTP e permitir interações via requests, execute:

```console
python app.py
```

Isso iniciará o servidor Flask, que estará disponível em `http://localhost:5000` (dependendo da configuração do seu ambiente).

## Exemplo de requisição

Requisição:

- Tipo: **POST**.
- URL: `http://localhost:5000/message`.
- Body **raw JSON**

Exemplo do Body:

```json
{
  "message": "Oi, tudo bem?"
}
```

Resposta:

```json
{
  "response": "Oi, que bom te ver aqui!"
}
```
