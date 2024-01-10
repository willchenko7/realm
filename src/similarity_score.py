from customTokenizer import loadCustomTokenizer
from csv_helpers import csv2lol
import math
import os

def vectorize(list1, list2):
    # Create a set of all unique elements
    union_set = set(list1) | set(list2)
    # Create vectors from the sets
    vec1 = [1 if item in list1 else 0 for item in union_set]
    vec2 = [1 if item in list2 else 0 for item in union_set]
    return vec1, vec2

def cosine_similarity(vec1, vec2):
    # Dot product
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    # Magnitude of vectors
    magnitude1 = math.sqrt(sum([v**2 for v in vec1]))
    magnitude2 = math.sqrt(sum([v**2 for v in vec2]))
    # Cosine similarity
    if magnitude1 * magnitude2 == 0:
        # To handle the case where one vector might be all zeros
        return 0
    else:
        return dot_product / (magnitude1 * magnitude2)

def get_scores(input_sentence,data,tokenizer):
    scores = []
    for sentence in data:
        x, y = vectorize(input_sentence,sentence)
        #get dictionary of token counts in input_sentence
        pad_token_id = tokenizer.pad_token_id
        x_dict = {i:input_sentence.count(i) for i in input_sentence if i != pad_token_id}
        #get max number of times a token appears in input_sentence
        x_max = max(x_dict.values())
        #print(x_max)
        if x_max > 5:
            scores.append(0)
            continue
        scores.append(cosine_similarity(x,y))
    return scores

def similiarity_score(input_sentence,tokenizer):
    if input_sentence == '[CLS][SEP]':
        return 0
    data_path = os.path.join("data","beyond_good_and_evil.csv")
    tokenizer_path = os.path.join("data","my_tokenizer")
    tokenizer,data_collator = loadCustomTokenizer(tokenizer_path)
    data = csv2lol(data_path)
    data = [" ".join(row) for row in data]
    data = [tokenizer.encode(row) for row in data]
    input_sentence = tokenizer.encode(input_sentence)
    scores = get_scores(input_sentence,data,tokenizer)
    return sum(scores)

if __name__ == '__main__':
    tokenizer_path = os.path.join("data","my_tokenizer")
    tokenizer,data_collator = loadCustomTokenizer(tokenizer_path)
    input_sentence = "[CLS]!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    print(similiarity_score(input_sentence,tokenizer))
