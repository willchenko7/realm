import numpy as np
from sim import sim
from interest_fitness import interest_fitness
import pickle
import os

'''
goal: use evolutionary algorithm to find optimal weights to maximize fitness

input:
    n_layers - number of layers in model
    input_size - size of input data
    layer_sizes - list of layer sizes
    pop_size - population size
    num_generations - number of generations
    num_parents - number of parents for crossover
    mutation_rate - mutation rate
    start_times - list of start times for simulation
    model_name - name of model (used for saving best solution)

output:
    saves solution as pickle file in models folder (models/best_solution_{model_name}.pkl)
'''

def mutate_initial_pop(solution, mutation_rate=0.01):
    w, b, attn_weights, attn_query, attn_keys, attn_values = solution
    mutated_w = [w_layer + np.random.randn(*w_layer.shape) * mutation_rate for w_layer in w]
    mutated_b = [b_layer + np.random.randn(*b_layer.shape) * mutation_rate for b_layer in b]
    mutated_attn_weights = attn_weights + np.random.randn(*attn_weights.shape) * mutation_rate
    mutated_attn_query = attn_query + np.random.randn(*attn_query.shape) * mutation_rate
    mutated_attn_keys = attn_keys + np.random.randn(*attn_keys.shape) * mutation_rate
    mutated_attn_values = attn_values + np.random.randn(*attn_values.shape) * mutation_rate
    return mutated_w, mutated_b, mutated_attn_weights, mutated_attn_query, mutated_attn_keys, mutated_attn_values

def initialize_population(pop_size, layer_sizes,from_file=None):
    '''
    goal: initialize population of size pop_size with random weights and biases for each layer
    -optional: initialize population based on best solution
    '''
    input_size = 1000
    if from_file is not None:
        #if we want to initialize population based on best solution
        population = []
        best_solution = pickle.load(open(os.path.join('models',from_file + '.pkl'),'rb'))
        for _ in range(pop_size):
            mutated_solution = mutate_initial_pop(best_solution)
            population.append(mutated_solution)
        return population
    population = []
    buffer = 1
    for _ in range(pop_size):
        w = [np.random.rand(input_size if i == 0 else layer_sizes[i - 1], size)*buffer for i, size in enumerate(layer_sizes)]
        b = [np.random.rand(size)*buffer for size in layer_sizes]
        attention_layer_index = 0 
        layer_output_dim = layer_sizes[attention_layer_index]
        attn_dim = layer_output_dim 
        attn_query = np.random.rand(attn_dim).astype(np.float64)
        attn_keys = np.random.rand(attn_dim, attn_dim).astype(np.float64)
        attn_values = np.random.rand(attn_dim, attn_dim).astype(np.float64)
        attn_weights = np.random.rand(attn_dim).astype(np.float64)
        population.append((w, b, attn_weights, attn_query, attn_keys, attn_values))
    return population

def compute_fitness(solution,fitness_type='interest',read_model_name=None,interest_mark=0.5):
    '''
    goal: compute fitness of solution
    -fitness is negative because we want to minimize the output
    '''
    if fitness_type == 'update_interest':
        fitness = interest_fitness(solution,read_model_name)
    else:
        fitness = -1*sim(solution,fitness_type=fitness_type,interest_mark=interest_mark)
    print(f"Final Fitness: {fitness}")
    return fitness  # Since the goal is to minimize the output

def select_parents(population, fitnesses, num_parents):
    '''
    goal: select parents for crossover
    -parents are selected based on fitness
    -the best num_parents solutions are selected as parents
    '''
    parents = list(np.argsort(fitnesses)[:num_parents])
    return [population[p] for p in parents]

def crossover(parent1, parent2):
    '''
    goal: perform crossover on parents
    -crossover is performed by taking the average of the weights and biases of the parents
    '''
    child_w = [(w1 + w2) / 2 for w1, w2 in zip(parent1[0], parent2[0])]
    child_b = [(b1 + b2) / 2 for b1, b2 in zip(parent1[1], parent2[1])]
    child_attn_weights = (parent1[2] + parent2[2]) / 2
    child_attn_query = (parent1[3] + parent2[3]) / 2
    child_attn_keys = (parent1[4] + parent2[4]) / 2
    child_attn_values = (parent1[5] + parent2[5]) / 2
    return child_w, child_b, child_attn_weights, child_attn_query, child_attn_keys, child_attn_values

def mutate(solution, mutation_rate):
    '''
    goal: mutate solution
    -mutation is performed by adding a random number to each weight and bias
    -adjust mutation rate to control how much mutation occurs
    '''
    w, b, attn_weights, attn_query, attn_keys, attn_values = solution
    mutated_w = [w_layer + np.random.randn(*w_layer.shape) * mutation_rate for w_layer in w]
    mutated_b = [b_layer + np.random.randn(*b_layer.shape) * mutation_rate for b_layer in b]
    mutated_attn_weights = attn_weights + np.random.randn(*attn_weights.shape) * mutation_rate
    mutated_attn_query = attn_query + np.random.randn(*attn_query.shape) * mutation_rate
    mutated_attn_keys = attn_keys + np.random.randn(*attn_keys.shape) * mutation_rate
    mutated_attn_values = attn_values + np.random.randn(*attn_values.shape) * mutation_rate
    return mutated_w, mutated_b, mutated_attn_weights, mutated_attn_query, mutated_attn_keys, mutated_attn_values

def ea(input_size,layer_sizes,pop_size,num_generations,num_parents,mutation_rate,model_name,read_model_name=None,fitness_type='interest',starting_point=None,interest_mark=0.5):
    '''
    goal: perform evolutionary algorithm
    '''
    # Initialize population
    if model_name == 'interest_model':
        population = initialize_population(pop_size, layer_sizes,from_file=model_name)
    else:
        if starting_point is None:
            population = initialize_population(pop_size, layer_sizes,None)
        else:
            population = initialize_population(pop_size, layer_sizes,from_file=starting_point)
    # Evolution
    for generation in range(num_generations):
        # Compute fitness for each solution
        fitnesses = np.array([compute_fitness(solution,fitness_type,read_model_name,interest_mark) for solution in population])
        # Select parents
        parents = select_parents(population, fitnesses, num_parents)
        # Generate next generation
        next_generation = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                child = crossover(parents[i], parents[i + 1])
                child = mutate(child, mutation_rate)
                next_generation.append(child)
        # Replace worst solutions with new ones
        worst_indices = np.argsort(fitnesses)[-len(next_generation):]
        for idx, new_sol in zip(worst_indices, next_generation):
            population[idx] = new_sol
        print(f"Generation {generation}: Best Fitness = {np.min(fitnesses)}")
    # Best solution
    best_index = np.argmin(fitnesses)
    best_solution = population[best_index]
    #print("Best Solution:", best_solution)
    best_fitness = fitnesses[best_index]
    print("Best Fitness:", best_fitness)
    #save best solution as pickle
    pickle.dump(best_solution, open(os.path.join('models',f'{model_name}.pkl'),'wb'))
    return best_fitness

if __name__ == "__main__":
    input_size = 1000
    layer_sizes = [500, 200, 100, 50, 1]
    pop_size = 10
    num_generations = 10
    num_parents = 10
    mutation_rate = 10
    #ea_type = 'update_interest'
    ea_type = ''
    if ea_type == 'update_interest':
        model_name = 'interest_model'
        read_model_name = 'model3'
        fitness_type = 'update_interest'
    else:
        model_name = 'model3'
        read_model_name = None
        fitness_type = 'interest'
    ea(input_size,layer_sizes,pop_size,num_generations,num_parents,mutation_rate,model_name,read_model_name,fitness_type)