'''
wrapper fn to make it easier to run different types of ea's
'''
from ea import ea

def run_ea(ea_type,input_model_name,starting_point):
    input_size = 1000
    layer_sizes = [500, 200, 100, 50, 1]
    pop_size = 50
    num_generations = 100
    num_parents = 50
    mutation_rate = 3
    #ea_type = 'update_interest'
    #ea_type = ''
    if ea_type == 'update_interest':
        model_name = 'interest_model'
        read_model_name = input_model_name
        fitness_type = 'update_interest'
    elif ea_type == 'similarity':
        model_name = input_model_name
        read_model_name = None
        fitness_type = 'similarity'
    elif ea_type == 'interest':
        model_name = input_model_name
        read_model_name = None
        fitness_type = 'interest'
    elif ea_type == 'diversity':
        model_name = input_model_name
        read_model_name = None
        fitness_type = 'diversity'
    else:
        model_name = input_model_name
        read_model_name = None
        fitness_type = 'interest'
    ea(input_size,layer_sizes,pop_size,num_generations,num_parents,mutation_rate,model_name,read_model_name,fitness_type,starting_point)
    return

if __name__ == '__main__':
    ea_type = 'diversity'
    input_model_name = 'model5'
    starting_point = 'model5'
    run_ea(ea_type,input_model_name,starting_point)