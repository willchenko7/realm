from generate import generate
from customTokenizer import loadCustomTokenizer
import os
from similarity_score import similiarity_score
import numpy as np

def sim(model):
    tokenizer_path = os.path.join("data","my_tokenizer")
    tokenizer,data_collator = loadCustomTokenizer(tokenizer_path)
    scores = []
    for _ in range(100):
        generated_text, x = generate(model,tokenizer)
        s = similiarity_score(generated_text,tokenizer)
        scores.append(s)
    return sum(scores)

if __name__ == '__main__':
    #randomly generate weights and biases
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
    print(sim(model))