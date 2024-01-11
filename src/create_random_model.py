'''
goal: create a model with random weights
'''
import numpy as np
import pickle
import os

def create_random_model(model_name):
    n_layers = 5
    input_size = 1000
    layer_sizes = [500, 200, 100, 50, 1]
    buffer = 1
    w = [np.random.rand(input_size if i == 0 else layer_sizes[i - 1], size)*buffer for i, size in enumerate(layer_sizes)]
    b = [np.random.rand(size)*buffer for size in layer_sizes]
    attention_layer_index = 0 
    layer_output_dim = layer_sizes[attention_layer_index]
    attn_dim = layer_output_dim 
    attn_query = np.random.rand(attn_dim).astype(np.float64)
    attn_keys = np.random.rand(attn_dim, attn_dim).astype(np.float64)
    attn_values = np.random.rand(attn_dim, attn_dim).astype(np.float64)
    attn_weights = np.random.rand(attn_dim).astype(np.float64)
    model = (w, b, attn_weights, attn_query, attn_keys, attn_values)
    #save model
    pickle.dump(model,open(os.path.join('models',model_name + '.pkl'),'wb'))
    return model

if __name__ == '__main__':
    model_name = 'interest_model'
    create_random_model(model_name)