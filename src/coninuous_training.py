from run_ea import run_ea
from generate_from_model import generate_from_model
from customTokenizer import loadCustomTokenizer
import os
from csv_helpers import lol2csv

'''
goal: coninuously train model, by steadily increasing the interest mark
    generate data each time it passes a threshold
'''

def continuous_training(starting_point,interest_mark,interest_mark_increment,fitness_threshold):
    tokenizer_path = os.path.join("data","my_tokenizer")
    tokenizer,data_collator = loadCustomTokenizer(tokenizer_path)
    ea_type = 'interest_torch'
    input_model_name = starting_point
    #interest_mark = 0.1
    #interest_mark_increment = 0.05
    #fitness_threshold = -75
    counter = 0
    while counter < 1000:
        counter += 1
        final_fitness = run_ea(ea_type,input_model_name,starting_point,interest_mark)
        if final_fitness < fitness_threshold:
            interest_mark += interest_mark_increment
            generated_text = generate_from_model(input_model_name + '.pkl',tokenizer)
            lol2csv(os.path.join("generated_data",input_model_name + str(interest_mark) + '.pkl'),generated_text)
    return
