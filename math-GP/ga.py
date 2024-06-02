import numpy as np

def sp_crossover(parent1, parent2, lb, ub):
    """
    Perform crossover between two parents using single-point crossover.

    Parameters:
    - parent1: The first parent.
    - parent2: The second parent.

    Returns:
    - offspring: The offspring.
    """
    while True:
        crossover_point = np.random.randint(0, len(parent1))
        offspring = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        if np.all((offspring >= lb) & (offspring <= ub)):
            break
    
    return offspring

def mutation(offspring, lb, ub):
    """
    Perform mutation on an offspring.
    
    Parameters:
    - offspring: The offspring to be mutated.
    
    Returns:
    - offspring: The mutated offspring.
    """
    
    while True:
        mutated_offspring = offspring + np.random.normal(0, 1, len(offspring))
        if np.all((mutated_offspring >= lb) & (mutated_offspring <= ub)):
            break
        
    return mutated_offspring

def genetic_algorithm(f, N, M, G, lb, ub, crossover, mutation):
    """
    Perform genetic algorithm optimization.

    Parameters:
    - f: The objective function to be minimized.
    - N: The number of individuals in the population.
    - M: The number of genes in each individual.
    - G: The number of generations.
    - lb: The lower bounds for the solution variables.
    - ub: The upper bounds for the solution variables.
    - crossover: The crossover function to be used.
    - mutation: The mutation function to be used.

    Returns:
    - X: The final best solution found.
    - X_list: A list of all solutions found during the optimization process.
    """
    
    # Initialize the population
    X = np.random.uniform(lb, ub, (N, M))
    X_list = X.tolist()
    
    # Evaluate the objective function for each individual 
    f_values = np.array([f(x) for x in X])
    
    for g in range(G):        
        # Select the best individuals
        best_idx = np.argsort(f_values)[:N//2]
        parents_values = f_values[best_idx]
        X = X[best_idx]
        
        # Crossover
        X_offspring = np.zeros((N - N//2, M))
        for i in range(N - N//2):
            parent1 = X[np.random.choice(N//2)]
            parent2 = X[np.random.choice(N//2)]
            X_offspring[i] = crossover(parent1, parent2, lb, ub)
        
        # Mutation
        for i in range(N - N//2):
            X_offspring[i] = mutation(X_offspring[i], lb, ub)
        
        offspring_values = np.array([f(x) for x in X_offspring])
        f_values = np.concatenate([parents_values, offspring_values])
        X = np.concatenate([X, X_offspring])
        X_list.extend(X_offspring)
    
    X_final = X[np.argmin(f(X))]
    return X_final, X_list

if __name__ == '__main__':
    """1 Dimensional Example"""
    
    f = lambda X: X[0]**2
    
    N = 100
    M = 1
    G = 100
    lb = -10
    ub = 10
    X, X_list = genetic_algorithm(f, N, M, G, lb, ub, sp_crossover, mutation)
    print(X)
    
    """2 Dimensional Example"""
    f = lambda X: X[0]**2 + X[1]**2
    N = 100
    M = 2
    G = 100
    lb = np.array([-10, -10])
    ub = np.array([10, 10])
    X, X_list = genetic_algorithm(f, N, M, G, lb, ub, sp_crossover, mutation)
    print(X)