import swarm_optimization
import numpy as np

# https://en.wikipedia.org/wiki/Test_functions_for_optimization is a good reference

    
# Basic function summing all components.  Used to test that constraint_function works.
class SumWithConstraint(swarm_optimization.OptimizationProblem):
    def __init__(self, max = 10, dim = 10):
        self.max = max
        self.param_dim = dim

    def initialize_particles(self, n: int):
        # Initialize particles with ranges outside our intended maximums
        return (np.random.rand(n, self.param_dim) * 4*self.max) - (2*self.max)

    def objective_function(self, particles):
        return np.sum(particles, axis=1).reshape(-1,1)
    
    def constraint_function(self, particles):
        return np.clip(particles, -self.max, self.max)
    
# Basic function summing all components.  Used to test that basic feasibility testing in objective function works
class SumWithFeasibility(swarm_optimization.OptimizationProblem):
    def __init__(self, max = 10, dim = 10):
        self.max = max
        self.param_dim = dim

    def initialize_particles(self, n: int):
        # Initialize particles with ranges outside our intended maximums
        return (np.random.rand(n, self.param_dim) * 4*self.max) - (2*self.max)

    def objective_function(self, particles):
        result = np.sum(particles, axis=1).reshape(-1,1)
        feasible = (particles < self.max).all(axis=1, keepdims=True)
        return result * feasible + ((1-feasible) * -np.finfo(np.float64).max)

class QuadraticFunction(swarm_optimization.OptimizationProblem):
    def __init__(self, A: np.ndarray, b: np.ndarray, c: np.ndarray):
        super().__init__()
        assert(len(A.shape) == 2)
        assert(A.shape[0] == A.shape[1])
        self.param_dim = A.shape[0]

        assert(len(b.shape) == 1)
        assert(b.shape[0] == self.param_dim)
        assert(len(c.shape) == 1)
        assert(c.shape[0] == 1)

        self.A = A.copy()
        self.b = b.copy()
        self.c = c.copy()
    
    def objective_function(self, particles):
        return ((particles @ self.A @ particles.transpose()).diagonal() + particles @ self.b + self.c).reshape(-1, 1)
# https://en.wikipedia.org/wiki/Ackley_function
class AckleyFunction(swarm_optimization.OptimizationProblem):
    def __init__(self, a=20, b=0.2, c = 2*np.pi, d = 3):
        self.a = a
        self.b = b
        self.c = c
        self.param_dim = d

    def initialize_particles(self, n: int):
        return (np.random.rand(n, self.param_dim) * 10) - 5

    def objective_function(self, particles):
        result = -self.a * np.exp(-self.b*np.sqrt(np.sum(particles**2, axis=1)/self.param_dim))
        result -= np.exp(np.sum(np.cos(self.c * particles), axis=1) / self.param_dim)
        return (- (result + self.a + np.e)).reshape(-1,1)
    
    def constraint_function(self, particles):
        return np.clip(particles, -5, 5)

# https://en.wikipedia.org/wiki/Rosenbrock_function
class RosenbrockFunction(swarm_optimization.OptimizationProblem):
    def __init__(self, a=1, b=100):
        self.a = a
        self.b = b
        self.param_dim = 2

    def initialize_particles(self, n: int):
        return (np.random.rand(n, self.param_dim) * 10) - 5

    def objective_function(self, particles):
        result = (self.a - particles[:, 0])**2 + self.b*(particles[:, 1] - particles[:, 0]**2)**2
        return -result.reshape(-1,1)
    
    def constraint_function(self, particles):
        return np.clip(particles, -5, 5)

# https://en.wikipedia.org/wiki/Rastrigin_function
class RastriginFunction(swarm_optimization.OptimizationProblem):
    def __init__(self, dim=5):
        self.A = 10
        self.param_dim = dim

    def initialize_particles(self, n: int):
        return (np.random.rand(n, self.param_dim) * 10.24) - 5.12

    def objective_function(self, particles):
        result = self.A * self.param_dim + np.sum(particles**2 - self.A * np.cos(2*np.pi*particles), axis=1)
        return result.reshape(-1,1)
    
    def constraint_function(self, particles):
        return np.clip(particles, -5.12, 5.12)