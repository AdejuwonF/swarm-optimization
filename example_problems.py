import swarm_optimization
import numpy as np

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

class AckleyFunction(swarm_optimization.OptimizationProblem):
    def __init__(self, a=20, b=0.2, c = 2*np.pi, d = 3):
        self.a = a
        self.b = b
        self.c = c
        self.param_dim = d
    def objective_function(self, particles):
        result = -self.a * np.exp(-self.b*np.sqrt(np.sum(particles**2, axis=1)/self.param_dim))
        result -= np.exp(np.sum(np.cos(self.c * particles), axis=1) / self.param_dim)
        return (- (result + self.a + np.e)).reshape(-1,1)
    
    def constraint_function(self, particles):
        return np.clip(particles, -5, 5)