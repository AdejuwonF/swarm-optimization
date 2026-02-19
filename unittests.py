import swarm_optimization
import example_problems
import unittest
import numpy as np

class SwarmOptimizationTestCase(unittest.TestCase):
    def test_ackley_optimization_2d(self):
        problem = example_problems.AckleyFunction(d=2)
        solver = swarm_optimization.ParticleSwarmSolver(problem=problem, momentum=0.7, 
                                                        global_attraction=1.5, personal_attraction=1.5, 
                                                        num_particles=50, v_max=0.5)
        solver.run(500)
        results = solver.get_result()
        self.assertTrue(np.isclose([0, 0], results[0]).all())
        self.assertTrue(np.isclose([0], results[1]).all())

    def test_ackley_optimization_10d(self):
        problem = example_problems.AckleyFunction(d=10)
        solver = swarm_optimization.ParticleSwarmSolver(problem=problem, momentum=0.7, 
                                                        global_attraction=1.5, personal_attraction=1.5, 
                                                        num_particles=2000, v_max=0.5)
        solver.run(5000)
        results = solver.get_result()
        self.assertTrue(np.isclose([0]*10, results[0]).all())
        self.assertTrue(np.isclose([0], results[1]).all())

    def test_rosenbrock_optimization_2d(self):
        problem = example_problems.RosenbrockFunction()
        solver = swarm_optimization.ParticleSwarmSolver(problem=problem, momentum=0.7, 
                                                        global_attraction=1.5, personal_attraction=1.5, 
                                                        num_particles=100, v_max=0.5)
        solver.run(1000)
        results = solver.get_result()
        self.assertTrue(np.isclose([1, 1], results[0]).all())
        self.assertTrue(np.isclose([0], results[1]).all())
    
    def test_sum_optimization_with_constraint5d(self):
        problem = example_problems.SumWithConstraint(max = 10, dim = 5)
        solver = swarm_optimization.ParticleSwarmSolver(problem=problem, momentum=0.7, 
                                                        global_attraction=1.5, personal_attraction=1.5, 
                                                        num_particles=2000, v_max=0.5)
        solver.run(10000)
        results = solver.get_result()
        self.assertTrue(np.isclose([10]*5, results[0]).all())
        self.assertTrue(np.isclose(10*5, results[1]).all())

    def test_sum_optimization_with_feasibility5d(self):
        problem = example_problems.SumWithFeasibility(max = 10, dim = 5)
        solver = swarm_optimization.ParticleSwarmSolver(problem=problem, momentum=0.7, 
                                                        global_attraction=1.5, personal_attraction=1.5, 
                                                        num_particles=2000, v_max=0.5)
        solver.run(10000)
        results = solver.get_result()
        self.assertTrue(np.isclose([10]*5, results[0]).all())
        self.assertTrue(np.isclose(10*5, results[1]).all())

    def test_rastrigin_optimization_5d(self):
        problem = example_problems.RastriginFunction(dim=5)
        solver = swarm_optimization.ParticleSwarmSolver(problem=problem, momentum=0.7, 
                                                        global_attraction=1.5, personal_attraction=1.5, 
                                                        num_particles=10000, v_max=0.5)
        solver.run(40000)
        results = solver.get_result()
        print(results)
        self.assertTrue(np.isclose([4.52299366]*5, np.abs(results[0])).all())
        self.assertTrue(np.isclose([201.7664509], results[1]).all())

if __name__ == "__main__":
    unittest.main()
