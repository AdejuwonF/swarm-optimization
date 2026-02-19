import numpy as np

class OptimizationProblem():

    def __init__(self):
        # Number of parameters our partciles optimize over
        self.param_dim = 0
    
    def initialize_particles(self, n: int):
        """
        Initialize an array of particles.  Defaults to uniform random distribution between [-1, 1]
        
        :param n: Number of particles to generate
        :type n: int
        :return: n x self.param_dim array of intialized particles.
        """
        return (np.random.rand(n, self.param_dim) * 2) - 1

    def initialize_particle_velocities(self, n: int):
        """
        Initialize an array of particle velocities.  Defaults to uniform random distribution between [-1, 1]
        
        :param n: Number of particles to generate
        :type n: int
        :return: n x self.param_dim array of intialized particles.
        """
        return (np.random.rand(n, self.param_dim) * 2) - 1

    def objective_function(self, particles: np.ndarray):
        """
        Assigns an objective score to each particle in an array.  
        Objective is assumed to be subject to maximization.
        Feasibility constraints (e.g if a child shape fits inside a parent shape)
        should be accounted for by setting the score to the largest negative value (-np.finfo(np.float64).max).

        :param particles: n x self.param_dim array of particles.
        :type particles: np.ndarray
        :return: n x 1 (for broadcasting convenience) dimension array of objective scores for each particle
        """
        raise Exception("ObjectiveNotImplemented")
    
    def constraint_function(self, particles: np.ndarray):
        """
        Takes in an array of particles and projects and component-wise constraints.
        For example, clipping values to be within a certain range or fixing a value to be constant.
        These should be simple constraints that apply to individual components.  Different than
        feasibility constraints that are applied in the objective function.
        By default it does nothing.
     
        :param particles: n x self.param_dim array of particles.
        :type particles: np.ndarray
        :return: n x self.param_dim array of particles with local constraints projected onto them.
        """
        return particles


class ParticleSwarmSolver():
    def __init__(self, problem: OptimizationProblem,
                 momentum: float = 0.8 ,
                 global_attraction  : float = 1.5,
                 personal_attraction : float = 1.5, 
                 num_particles : int = 1000,
                 v_max : float = 0.5):
        """
        Docstring for __init__
        
        :param self: Description
        :param problem: Description
        :type problem: OptimizationProblem
        :param momentum: Description
        :type momentum: float
        :param global_attraction: Description
        :type global_attraction: float
        :param personal_attraction: Description
        :type personal_attraction: float
        :param num_particles: Description
        :type num_particles: int
        """
        self.problem = problem
        self.momentum = momentum
        self.global_attraction = global_attraction
        self.personal_attraction = personal_attraction
        self.num_particles = num_particles
        self.v_max = v_max

        self.positions = self.problem.initialize_particles(self.num_particles)
        self.positions = self.problem.constraint_function(self.positions)
        self.velocities = self.problem.initialize_particle_velocities(self.num_particles)
        self.velocities = np.clip(self.velocities, -self.v_max, self.v_max)
        self.scores = self.problem.objective_function(self.positions)
        self.best_scores = self.scores.copy()
        self.best_positions = self.positions.copy()
        self.best_score_index = np.argmax(self.best_scores)

        # print("score", self.best_scores[self.best_score_index])
        # print("positions", self.positions)
        # print("velocities", self.velocities)
        # print("best positions", self.best_positions)
    
    def run(self, max_iters = 1000, epislon=-1):
        for i in range(max_iters):
            self.velocities = self.momentum * self.velocities \
            + (self.personal_attraction * np.random.rand()) * (self.best_positions - self.positions) \
            + (self.global_attraction * np.random.rand()) * (self.best_positions[self.best_score_index] - self.positions)
            # print("momentum", self.momentum * self.velocities )
            # print("personal", (self.personal_attraction * np.random.rand()) * (self.best_positions - self.positions))
            # print("global", (self.global_attraction * np.random.rand()) * (self.best_positions[self.best_score_index] - self.positions))
            self.velocities = np.clip(self.velocities, -self.v_max, self.v_max)

            self.positions += self.velocities
            self.positions = self.problem.constraint_function(self.positions)
            self.scores = self.problem.objective_function(self.positions)

            old_best_score = self.best_scores[self.best_score_index]
            score_comparision = (self.scores > self.best_scores).astype(float).reshape(-1, 1)
            # print("score_comparisons", score_comparision)
            # print("best_scores", self.best_scores)
            # print("scores", self.scores)
            self.best_scores = score_comparision * self.scores + (1 - score_comparision) * self.best_scores
            self.best_positions = score_comparision * self.positions + (1 - score_comparision) * self.best_positions
            self.best_score_index = np.argmax(self.best_scores)
            # print("updated_scores", self.best_scores)

            best_score = self.best_scores[self.best_score_index]
            # print("score", best_score)
            # print("positions", self.positions)
            # print("velocities", self.velocities)
            # print("best positions", self.best_positions)
            score_diff = best_score - old_best_score
            if score_diff < epislon:
                break
    
    def get_result(self):
        return (self.best_positions[self.best_score_index],  self.best_scores[self.best_score_index])

    
        
    
