import numpy as np
from tqdm import tqdm

def T(k):
    """Temperature function"""
    return 100*(0.8**k)

def P(delta, T):
    """Acceptance probability function"""
    return np.exp(-delta/T)

def simulated_annealing(f, X0, T, P, N, lb, ub):
    """
    Perform simulated annealing optimization.

    Parameters:
    - f: The objective function to be minimized.
    - X0: The initial solution. Please note that this should be a numpy array or a scalar.
    - T: A function that determines the temperature at each iteration.
    - P: A function that determines the probability of accepting a worse solution.
    - N: The number of iterations.
    - lb: The lower bounds for the solution variables.
    - ub: The upper bounds for the solution variables.

    Returns:
    - X: The final solution.
    - X_list: A list of all solutions found during the optimization process.
    """
    
    X = X0 if type(X0) is np.ndarray else np.array([X0])
    X_list = [X]
    
    for k in tqdm(range(N)):
        while True:
            X_new = X + np.random.normal(0, T(k), len(X))
            if np.all(X_new >= lb) and np.all(X_new <= ub):
                break
        delta = f(X_new) - f(X)
        if delta < 0 or np.random.uniform() < P(delta, T(k)):
            X = X_new
            X_list.append(X)
            
    return X, X_list

if __name__ == "__main__":
    # Example usage of 1 dimensional simulated annealing
    print("1D Example")
    f = lambda x: x[0]**2
    X0 = np.array([10])
    T = lambda k: 100*(0.8**k)
    P = lambda delta, T: np.exp(-delta/T)
    N = 1000
    lb = np.array([-10])
    ub = np.array([10])
    X, X_list = simulated_annealing(f, X0, T, P, N, lb, ub)
    
    print("Final solution:", X)
    
    # Example usage of 2 dimensional simulated annealing
    print("\n2D Example")
    f = lambda X: np.sum(X[0]**2 + X[1])
    X0 = np.array([10, 10])
    T = lambda k: 100*(0.8**k)
    P = lambda delta, T: np.exp(-delta/T)
    N = 1000
    lb = np.array([-10, -10])
    ub = np.array([10, 10])
    X, X_list = simulated_annealing(f, X0, T, P, N, lb, ub)
    
    print("Final solution:", X)