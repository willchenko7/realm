

def determine_diversity(tokenized_sentences):
    #get all unique tokens in a list of tokenized sentences
    unique_tokens = []
    for sentence in tokenized_sentences:
        for token in sentence:
            if token not in unique_tokens:
                unique_tokens.append(token)
    return len(unique_tokens)