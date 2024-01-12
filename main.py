
'''
0. create models with random weights
loop:
    1. ea ('')
    2. generate_from_model(model_name)
    3. prepare_for_label(model_name)
    4. HUMAN LABELING
    5. ea('update_interest')
    repeat
'''
import sys
import os
sys.path.append('src')
from src.run_ea import run_ea
from src.generate_from_model import generate_from_model
from src.prepare_for_label import prepare_for_label
from src.customTokenizer import loadCustomTokenizer

def main(model_name):
    #load tokenizer
    tokenizer_path = os.path.join("data","my_tokenizer")
    tokenizer,data_collator = loadCustomTokenizer(tokenizer_path)
    run_ea('update_interest',model_name,None)
    run_ea('',model_name,None)
    generate_from_model(model_name + '.pkl',tokenizer)
    prepare_for_label(model_name + '.csv')
    print('The program has finished running. Please label the generated data in the generated_data folder and run the program again.')
    return

if __name__ == '__main__':
    model_name = 'model5'
    main(model_name)