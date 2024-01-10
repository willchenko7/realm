import numpy as np

'''
goal: forward pass through neural network
- two options: forward pass with attention and forward pass without attention
'''

#different activation functions

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

#forward pass without attention
def forward(x, w, b, n_layers):
    buffer = 1
    x = x.astype(np.float64)
    w = [iw.astype(np.float64) for iw in w]
    b = [ib.astype(np.float64) for ib in b]
    for i in range(n_layers):
        x = np.dot(x, w[i]*buffer) + b[i]*buffer
    y = np.sum(x)/1e10
    y = sigmoid(y)
    return y

def attention(weights, query, keys, values):
    # Simple dot-product attention
    attn_scores = np.dot(query, keys.T)
    attn_distribution = softmax(attn_scores)
    output = np.dot(attn_distribution, values)
    return output * weights

def forward_with_attention(x, w, b, n_layers, attn_weights, attn_query, attn_keys, attn_values):
    x = x.astype(np.float64)
    w = [iw.astype(np.float64) for iw in w]
    b = [ib.astype(np.float64) for ib in b]

    for i in range(n_layers):
        x = np.dot(x, w[i]) + b[i]

        # Add attention mechanism at a specific layer
        if i == 0:
            x = attention(attn_weights, attn_query, attn_keys, x)

    y = np.sum(x) / 1e14
    #print(f'y: {y}')
    y = sigmoid(y)
    return y

if __name__ == "__main__":
    from datetime import datetime
    np.random.seed(0)
    x = np.random.rand(1000)

    n_layers = 5
    input_size = 1000
    layer_sizes = [500, 200, 100, 50, 1]

    np.random.seed(None)
    # Initialize weights and biases
    w = [np.random.rand(input_size if i == 0 else layer_sizes[i - 1], size) for i, size in enumerate(layer_sizes)]
    b = [np.random.rand(size) for size in layer_sizes]

    attention_layer_index = 0  # Index of the layer after which you want to apply attention

    # Dimension of the layer output where attention is applied
    layer_output_dim = layer_sizes[attention_layer_index]

    # Initialize attention weights
    # Assuming the query, keys, and values have the same dimension for simplicity
    attn_dim = layer_output_dim  # You can choose a different dimension if needed

    attn_query = np.random.rand(attn_dim).astype(np.float64)
    attn_keys = np.random.rand(attn_dim, attn_dim).astype(np.float64)
    attn_values = np.random.rand(attn_dim, attn_dim).astype(np.float64)
    attn_weights = np.random.rand(attn_dim).astype(np.float64)

    # Get the output
    stopwatch = datetime.now()
    #output = forward(x, w, b, n_layers)
    output = forward_with_attention(x, w, b, n_layers, attn_weights, attn_query, attn_keys, attn_values)
    runtime = datetime.now() - stopwatch
    print(output)
    print(f"Runtime: {runtime}")
