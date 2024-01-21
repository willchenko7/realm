'''
Goal: Given a sentence, determine whether it is interesting or not.

using the interest_model.pt file, we can load the model and use it to predict whether a sentence is interesting or not.
'''
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from collections import Counter, OrderedDict
from torchtext.vocab import Vocab
import re
from csv_helpers import csv2lol
from torch.nn.utils.rnn import pad_sequence
from interestTorchModel import interestTorchModel
import pickle


def preprocess_text(text, tokenizer, vocab, max_length):
    #text = text.lower()
    text = text.replace('[CLS]',' [CLS] ')
    text = text.replace('[SEP]',' [SEP] ' )
    tokens = tokenizer(text)
    #print(f'tokens: {tokens}')
    #token_indices = [vocab[token] for token in tokens]
    token_indices = []
    for token in tokens:
        if token in vocab:
            token_index = vocab[token]
            #print(f'"{token}" in vocab, index: {token_index}')
            token_indices.append(token_index)
        else:
            #print(f'"{token}" not in vocab')
            token_indices.append(vocab['<unk>'])
    #print(f'token_indices: {token_indices}')
    # Pad or truncate to max_length
    padded_indices = token_indices[:max_length] + [vocab['<pad>']] * (max_length - len(token_indices))
    return torch.tensor(padded_indices, dtype=torch.int64)

def predict_sentiment(model, sentence, tokenizer, vocab, max_length):
    model.eval()  # Set the model to evaluation mode
    processed_text = preprocess_text(sentence, tokenizer, vocab, max_length).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():  # Disable gradient calculation
        prediction = model(processed_text)
        # Apply softmax if your model's output is logits
        probabilities = torch.softmax(prediction, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class.item(), probabilities.squeeze().tolist()

def determine_interest_from_torch(sentence, interest_mark=0.5):
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    #print(f'vocab_size: {vocab_size}')
    #vocab_size = 17491
    emsize = 64
    num_class = 2
    model = interestTorchModel(vocab_size, emsize, num_class)
    model.load_state_dict(torch.load('models/interest_model.pth'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    max_length = 50  # Choose an appropriate max length
    tokenizer = get_tokenizer('basic_english')
    predicted_class, probabilities = predict_sentiment(model, sentence, tokenizer, vocab, max_length)
    #print(f"Predicted class: {predicted_class}, Probabilities: {probabilities}")
    #interest_mark = 0.01
    if probabilities[1] > interest_mark:
        predicted_class = 1
    else:
        predicted_class = 0
    return predicted_class

if __name__ == '__main__':
    #sentence = "The fact of the matter occurade around you in your life and I’m doing the people in the world"
    sentences = [
        "This is a sentence.",
        ]
    for sentence in sentences:
        interest = determine_interest_from_torch(sentence)
        print(f'interest: {interest}')
