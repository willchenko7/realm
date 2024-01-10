from forward import forward_with_attention
import numpy as np

def generate(model,tokenizer):
    w, b, attn_weights, attn_query, attn_keys, attn_values = model
    pad_token_id = tokenizer.pad_token_id
    sep_token_id = tokenizer.sep_token_id
    cls_token_id = tokenizer.cls_token_id
    input_size = 1000
    x = np.array([pad_token_id for _ in range(input_size)])
    #replace last token with cls_token_id
    x[-1] = cls_token_id
    for _ in range(100):
        y = forward_with_attention(x, w, b, 5, attn_weights, attn_query, attn_keys, attn_values)
        #if y is Nan, replace it with 0
        if np.isnan(y):
            y = 0
        #map y (a float between 0 and 1) to a token id in the range [0,sep_token_id]
        y = int(y * sep_token_id)
        #add the token to the end of the input, dropping the first token
        x = np.append(x[1:],y)
        if y == sep_token_id:
            break
    #decode the generated tokens into a string. do not include padding tokens
    generated_text = tokenizer.decode(x[x != pad_token_id])
    return generated_text, x

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
    #load tokenizer
    from customTokenizer import loadCustomTokenizer
    tokenizer_path = 'data/my_tokenizer'
    tokenizer,data_collator = loadCustomTokenizer(tokenizer_path)
    #generate text
    generated_text, x = generate(model,tokenizer)
    print(generated_text)
