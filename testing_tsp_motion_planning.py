from gcs_for_blocks.motion_planning_program import MotionPlanning
from gcs_for_blocks.axis_aligned_set_tesselation import (
    Box,
    AlignedSet,
    plot_list_of_aligned_sets,
    locations_to_aligned_sets,
    axis_aligned_tesselation,
)
import numpy as np

from pydrake.solvers import MathematicalProgram, Solve
from pydrake.math import le, eq
from gcs_for_blocks.tsp_solver import Vertex, Edge
from gcs_for_blocks.util import timeit, INFO, WARN, ERROR, YAY


bounding_box = AlignedSet(b=0, a=12, l=0, r=12)
block_width = 1

start = [(0, 0), (1, 1), (3, 5), (7, 4)]
target = [(0, 0), (5, 11), (9, 7), (5, 8)]
start_block_pos = start[1:]
target_block_pos = target[1:]
num_blocks = len(start_block_pos)
# get obstacles
obstacles = locations_to_aligned_sets(start_block_pos, target_block_pos, block_width)
# make a tesselation
convex_sets = axis_aligned_tesselation(bounding_box.copy(), obstacles)
convex_relaxation = False
moving_block_index = 0
smbi = str(moving_block_index)


visitations = np.array([0, 1, 1])  # will become mute
# simulating the thing
prog = MathematicalProgram()
vertices = dict()
edges = dict()
start_tsp = "s" + smbi + "_tsp"
target_tsp = "t" + smbi + "_tsp"
vertices[start_tsp] = Vertex(start_tsp)
vertices[target_tsp] = Vertex(target_tsp)

####################################
# add variables to start and target vertices
# associated vaiables are visitations, n x 1, each 0 or 1
vertices[start_tsp].set_v(prog.NewContinuousVariables(num_blocks, "visit_" + start_tsp))
vertices[target_tsp].set_v(
    prog.NewContinuousVariables(num_blocks, "visit_" + target_tsp)
)
# visitation constraints
visitation_box = Box(
    lb=np.zeros(num_blocks), ub=np.ones(num_blocks), state_dim=num_blocks
)
vA, vb = visitation_box.get_hpolyhedron()
v = vertices[start_tsp]
prog.AddLinearConstraint(le(vA @ v.v, vb))
prog.AddLinearConstraint(eq(v.v, visitations))
v = vertices[target_tsp]
prog.AddLinearConstraint(le(vA @ v.v, vb))
prog.AddLinearConstraint(eq(v.v, vertices[start_tsp].v))

MotionPlanning(
    prog,
    vertices,
    edges,
    bounding_box,
    start_block_pos,
    target_block_pos,
    convex_sets,
    moving_block_index,
    convex_relaxation,
)

primal_solution = Solve(prog)
if primal_solution.is_success():
    YAY("Optimal primal cost is %.5f" % primal_solution.get_optimal_cost())
else:
    ERROR("PRIMAL SOLVE FAILED!")
    ERROR("Optimal primal cost is %.5f" % primal_solution.get_optimal_cost())
    raise Exception

flows = [primal_solution.GetSolution(e.phi) for e in edges.values()]
not_tight = np.any(np.logical_and(0.01 < np.array(flows), np.array(flows) < 0.99))
if not_tight:
    WARN("CONVEX RELAXATION NOT TIGHT")
else:
    YAY("CONVEX RELAXATION IS TIGHT")


def find_path_to_target(edges, start):
    """Given a set of active edges, find a path from start to target"""
    edges_out = [e for e in edges if e.left == start]
    assert len(edges_out) == 1
    current_edge = edges_out[0]
    v = current_edge.right

    target_reached = v.name == target_tsp

    if target_reached:
        return [start] + [v], [current_edge]
    else:
        v, e = find_path_to_target(edges, v)
        return [start] + v, [current_edge] + e


flow_vars = [(e, primal_solution.GetSolution(e.phi)) for e in edges.values()]
non_zero_edges = [e for (e, flow) in flow_vars if flow > 0.01]
v_path, e_path = find_path_to_target(non_zero_edges, vertices[start_tsp])
loc_path = [primal_solution.GetSolution(e.right_pos) for e in e_path]
loc_path[0] = primal_solution.GetSolution(e_path[1].left_pos)

plot_list_of_aligned_sets(
    convex_sets, bounding_box, visitations, moving_block_index, loc_path
)
plot_list_of_aligned_sets(
    start + target, tesselation, bounding_box, block_width_minus_delta
)
