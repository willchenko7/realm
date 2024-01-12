from generate import generate
from customTokenizer import loadCustomTokenizer
import os
from similarity_score import similiarity_score
from determine_interest import determine_interest
from determine_diversity import determine_diversity
import numpy as np

def fitness(generated_text,x,tokenizer,fitness_type='interest'):
    if fitness_type == 'interest':
        score = determine_interest(x)
    elif fitness_type == 'similarity':
        score = similiarity_score(generated_text,tokenizer)
    else:
        raise ValueError("fitness_type must be 'interest' or 'similarity'")
    return score

def sim(model,fitness_type='interest'):
    tokenizer_path = os.path.join("data","my_tokenizer")
    tokenizer,data_collator = loadCustomTokenizer(tokenizer_path)
    scores = []
    all_generated_text = []
    all_x = []
    for _ in range(100):
        generated_text, x = generate(model,tokenizer)
        all_generated_text.append(generated_text)
        all_x.append(x)
    
    if fitness_type == 'diversity':
        return determine_diversity(all_x)

    for generated_text,x in zip(all_generated_text,all_x):
        if generated_text == '[CLS][SEP]':
            score = 0
        else:
            pad_token_id = tokenizer.pad_token_id
            x_dict = {i:list(x).count(i) for i in list(x) if i != pad_token_id}
            x_max = max(x_dict.values())
            if x_max > 5:
                score = 0
            else:
                score = fitness(generated_text,x,tokenizer,fitness_type='interest')
        #s = similiarity_score(generated_text,tokenizer)
        scores.append(score)
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