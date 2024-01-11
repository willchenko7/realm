

def determine_diversity(tokenized_sentence):
    #get the number of unique tokens in the sentence
    unique_tokens = set(tokenized_sentence)
    return len(unique_tokens)