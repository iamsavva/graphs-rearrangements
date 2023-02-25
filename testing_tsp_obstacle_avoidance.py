import numpy as np

from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module
    MathematicalProgram,
    Solve,
)
from pydrake.math import le, eq  # pylint: disable=import-error, no-name-in-module
from gcs_for_blocks.tsp_vertex_edge import Vertex, Edge
from gcs_for_blocks.util import timeit, INFO, WARN, ERROR, YAY
from gcs_for_blocks.tsp_obstacle_avoidance import BlockMovingObstacleAvoidance
from gcs_for_blocks.motion_planning_obstacles_on_off import MotionPlanning
from gcs_for_blocks.axis_aligned_set_tesselation import (
    Box,
    AlignedSet,
    plot_list_of_aligned_sets,
    locations_to_aligned_sets,
    axis_aligned_tesselation,
)
from draw_2d import Draw2DSolution

#############################################################################


def process_and_solve(
    bounding_box, start, target, block_width, convex_relaxation=False, fast=True
):
    ##################
    # tolerances
    # small tolerance: each set will be offset inwards by this amount to ensure non-empty interior
    # for the scenarios in which blocks touch each other
    # for interior-point-method's sake, this is better than keeping edges with no interior
    # + set tesselation becomes easier
    set_tol = 0.00001
    # offset the bounding box inwards
    half_block_width = block_width / 2
    half_block_width_minus_tol = half_block_width - set_tol
    # NOTE: we offset the bounding box by half-block-width, else we go through walls
    bounding_box.offset_in(half_block_width_minus_tol)
    # compute offset block width and the tolerance for what's to be considered a shared edge
    block_width_minus_tol = block_width - set_tol
    share_edge_tol = set_tol / 50
    x = timeit()
    ##################
    # define the program
    prog = BlockMovingObstacleAvoidance(
        start_pos=start,
        target_pos=target,
        bounding_box=bounding_box,
        block_width=block_width_minus_tol,
        convex_relaxation=convex_relaxation,
        share_edge_tol=share_edge_tol,
    )
    x.dt("Building the program")
    # solve
    prog.solve()

    positions, modes = prog.get_trajectory_for_drawing()
    # draw
    bounding_box.offset_in(-half_block_width_minus_tol)
    target_position = prog.target_pos.copy()
    target_position.resize(target_position.size)
    drawer = Draw2DSolution(
        prog.num_blocks + 1,
        np.array([bounding_box.r, bounding_box.a]),
        modes,
        positions,
        target_position,
        fast=fast,
        no_arm=False,
        no_padding=True,
    )
    drawer.draw_solution()


#############################################################################
# examples
#############################################################################
block_width = 1

#############################################################################
# example 1
bounding_box = AlignedSet(b=0, a=6, l=0, r=7)
start = [
    (0.5, 4.5),
    (0.5, 0.5),
    (0.5, 2.5),
    (2.5, 2.5),
    (2.5, 0.5),
    (0.5, 5.5),
    (2.5, 4.5),
]
target = [
    (0.5, 5.5),
    (6.5, 0.5),
    (4.5, 0.5),
    (4.5, 2.5),
    (4.5, 4.5),
    (6.5, 4.5),
    (6.5, 2.5),
]
fast = True

# #############################################################################
# # example 2
# bounding_box = AlignedSet(b=0, a=3, l=0, r=5)
# start = [(2.5, 1), (1.5, 0.5), (0.5, 1.5), (2 - 0.5, 2 - 0.5), (2 - 0.5, 1 - 0.5)]
# target = [(2.5, 1), (4.5, 0.5), (4.5, 1.5), (4 - 0.5, 2 - 0.5), (4 - 0.5, 1 - 0.5)]

# #############################################################################
# # example 3
# bounding_box = AlignedSet(b=0, a=2, l=0, r=5)
# start = [(2.5, 1), (0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)]
# target = [(2.5, 1), (4.5, 0.5), (4.5, 1.5), (3.5, 1.5), (3.5, 0.5)]
# #############################################################################
# # example 4
# bounding_box = AlignedSet(b=0, a=3, l=0, r=3)
# start = [(2.25, 1.5), (0.5, 0.5), (2.25, 1.5)]
# target = [(2, 1), (2.5, 2.5), (2.25, 1.5)]
# #############################################################################
# # example 5, random
np.random.seed(1)
nb = 10
ub = 20
bounding_box = AlignedSet(b=0, a=ub, l=0, r=ub)
start = [
    tuple(np.random.uniform(0 + block_width / 2, ub - block_width / 2, 2))
    for i in range(nb + 1)
]
target = [
    tuple(np.random.uniform(0 + block_width / 2, ub - block_width / 2, 2))
    for i in range(nb + 1)
]

convex_relaxation = True


#############################################################################
process_and_solve(
    bounding_box=bounding_box,
    start=start,
    target=target,
    block_width=block_width,
    convex_relaxation=convex_relaxation,
    fast=fast,
)
