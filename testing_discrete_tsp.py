from gcs_for_blocks.tsp_solver import build_block_moving_gcs_tsp, TSPasGCS
import numpy as np
from gcs_for_blocks.util import timeit

# why do cycles occur?
# np.random.seed(1)
# some random seeds break symmetries; figure out what those symmetries are!!!
nb = 20
start = np.random.uniform(0, 50, nb + 1)
target = np.random.uniform(0, 50, nb + 1)
# start = np.array([(0,), (3,), (15,), (3,), (4,), (5,), (6,), (7,), (8,)]) + np.random.uniform(0.001, 0.002, 9)
# target = np.array([(19,), (2,), (4,), (5,), (23,), (7,), (30,), (9,), (11,)])


# start = np.random.uniform(0,50,30)
# target = np.random.uniform(0,50,30)

# start = np.array([0.0, 1, 2, 3, 4, 5, 6, 7, 10, 20]) + np.random.normal(0,0.001, nb+1)
# target = np.array([2.0, 2, 4, 5, 6, 7, 30, 9, 11, 12]) + np.random.normal(0,0.001, nb+1)

# nb = 4
# start = np.array([0.0, 1, 5, 6, 7])
# target = np.array([2.0, 3, 7, 8, 9])

# nb = 4
# start = np.array([0.0, 1, 2, 3, 4])
# target = np.array([2.0, 3, 4, 5, 6])

# nb = 2
# start = np.array([0.0, 1, 2])
# target = np.array([2.0, 3, 4])

# randomness does not break the cycles
# np.random.seed(2)
# start += np.random.normal(0, 0.01, len(start))
# target += np.random.normal(0, 0.01, len(target))
# convex_relaxation = False
convex_relaxation = False

x = timeit()
tsp = build_block_moving_gcs_tsp(start, target)
tsp.build_primal_optimization_program(convex_relaxation)
x.dt("Building the program")
tsp.solve_primal()
# tsp.verbose_solution()

# d.build_from_start_and_target(start, target)
# d.build_dual_optimization_program()

# # d.build_primal_optimization_program()
# # d.solve(convex_relaxation)
