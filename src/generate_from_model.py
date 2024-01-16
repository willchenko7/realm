from generate import generate
from customTokenizer import loadCustomTokenizer
import os
from csv_helpers import lol2csv
import pickle

def generate_from_model(model_name,tokenizer):
    model = pickle.load(open(os.path.join('models',model_name),'rb'))
    all_generated_text = []
    for _ in range(100):
        generated_text, x = generate(model,tokenizer)
        all_generated_text.append([generated_text])
    return all_generated_text

if __name__ == '__main__':
    model_name = "model6.pkl"
    #load tokenizer
    tokenizer_path = os.path.join("data","my_tokenizer")
    tokenizer,data_collator = loadCustomTokenizer(tokenizer_path)
    
    #generate text
    generated_text = generate_from_model(model_name,tokenizer)
    print(generated_text)
    #save generated text to csv
    lol2csv(os.path.join("generated_data",model_name.replace('.pkl','.csv')),generated_text)