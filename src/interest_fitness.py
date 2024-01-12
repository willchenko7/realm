'''
goal: determine fitness of interest model based on latest feedback
'''
import numpy as np
import pickle
import os
from forward import interest_forward
from csv_helpers import csv2lol
from customTokenizer import loadCustomTokenizer

def interest_fitness(interest_model,model_name):
    tokenizer_path = os.path.join("data","my_tokenizer")
    tokenizer,data_collator = loadCustomTokenizer(tokenizer_path)
    w, b, attn_weights, attn_query, attn_keys, attn_values = interest_model
    #load latest feedback
    feedback = csv2lol(os.path.join('labeled_data',model_name + '.csv'))
    #label is in first column, sentence is in second column
    total_error = 0
    for label, sentence in feedback:
        #tokenize sentence
        tokenized_sentence = tokenizer.encode(sentence)
        pad_token_id = tokenizer.pad_token_id
        input_size = 1000
        x = np.array([pad_token_id for _ in range(input_size)])
        #replace last n tokens with tokenized sentence
        x[-len(tokenized_sentence):] = tokenized_sentence
        #forward pass
        output = interest_forward(x, w, b, 5, attn_weights, attn_query, attn_keys, attn_values)
        #calculate error
        error = abs(float(label) - float(output))
        total_error += error
    if np.isnan(total_error):
        total_error = 10000
    return total_error

if __name__ == '__main__':
    model_name = 'model3'
    model = pickle.load(open(os.path.join('models','interest_model' + '.pkl'),'rb'))
    fitness = interest_fitness(model,model_name)
    print(fitness)