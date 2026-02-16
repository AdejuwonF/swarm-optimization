import swarm_optimization
import example_problems
import unittest
import numpy as np

class SwarmOptimizationTestCase(unittest.TestCase):
    def test_ackley_optimization(self):
        problem = example_problems.AckleyFunction(d=2)
        solver = swarm_optimization.ParticleSwarmSolver(problem=problem, momentum=0.7, 
                                                        global_attraction=1.5, personal_attraction=1.5, 
                                                        num_particles=50, v_max=0.5)
        solver.run(500)
        results = solver.get_result()
        self.assertTrue(np.isclose([0, 0], results[0]).all())
        self.assertTrue(np.isclose([0], results[1]).all())

if __name__ == "__main__":
    unittest.main()
