'''
goal: create a moel that will determine the interest of a user
- starts off random
- updates weights each time it recieves feedback from the user

'''
import numpy as np
import pickle
import os
from forward import interest_forward

def determine_interest(tokenized_sentence):
    model_name = 'interest_model'
    model = pickle.load(open(os.path.join('models',model_name + '.pkl'),'rb'))
    w, b, attn_weights, attn_query, attn_keys, attn_values = model
    output = interest_forward(tokenized_sentence, w, b, 5, attn_weights, attn_query, attn_keys, attn_values)
    return output