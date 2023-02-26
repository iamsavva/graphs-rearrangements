import typing as T

import numpy as np
import numpy.typing as npt

from pydrake.solvers import (
    MathematicalProgram,
    Solve,
)  # pylint: disable=import-error, no-name-in-module
from pydrake.math import le, eq  # pylint: disable=import-error, no-name-in-module
import pydot

from util import timeit, INFO, YAY, WARN, ERROR

from vertex import Vertex, VertexTSPprogram, VertexAlignedSet, VertexTSP
from edge import Edge, EdgeMotionPlanningProgam
from graph_tsp_gcs import ProgramOptionsForGCSTSP
from axis_aligned_set import AlignedSet, Box, FREE
from axis_aligned_set_tesselation import (
    AxisAlignedSetTessellation,
    make_a_test_with_objects_and_obstacles,
    HALF_BLOCK_WIDTH,
)
from axis_aligned_graph import GraphOfAdjacentAlignedSets

from tsp_gcs_program import GraphTSPGCSProgram

from graph_tsp_gcs import GraphTSPGCS


import pydrake.geometry.optimization as opt  # pylint: disable=import-error, no-name-in-module
from pydrake.geometry.optimization import (  # pylint: disable=import-error, no-name-in-module
    Point,
    GraphOfConvexSets,
    HPolyhedron,
    ConvexSet,
)
from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module, unused-import
    Binding,
    L2NormCost,
    LinearConstraint,
    LinearEqualityConstraint,
    LinearCost,
)


class DescentOverPermutahedron:
    """
    Collision-free motion planing as a shortest path GCS.
    Say we have n blocks, where each can be either in a start or in a goal position.
    We can formulate the problem as collision free motion planning with 2*n obstacles with some of
    the obstacles turn on, and some -- off. This allows us to fix the 2d space tesselation, and then
    do shortest path MP for any box i, given the information on whether the other boxes are in the
    start or target positions. Visitation vector captures whether box i is in a start position
    (visitation[i] = 0), or in a goal position (visitation[i] = 1).
    Depending on that value, we can turn on or off that obstacle.
    """

    def __init__(
        self,
        num_objects:int,
        arm_start: npt.NDArray,
        arm_target: npt.NDArray,
        start_object_locations: npt.NDArray,
        target_object_locations: npt.NDArray,
        tessellation_graph: GraphOfAdjacentAlignedSets,
    ) -> None:
        self.num_gcs_solves = 0

        self.tessellation_graph = tessellation_graph

        self.num_objects = num_objects
        self.start_object_locations = np.array(start_object_locations)
        self.target_object_locations = np.array(target_object_locations)
        assert num_objects == len(self.start_object_locations)
        assert num_objects == len(self.target_object_locations)

        self.arm_start_to_si = [ np.linalg.norm(arm_start-self.start_object_locations[i]) for i in range(self.num_objects) ]
        self.ti_to_arm_target = [ np.linalg.norm(arm_target-self.target_object_locations[i]) for i in range(self.num_objects) ]

        self.ti_to_sj = np.array([ [np.linalg.norm(self.target_object_locations[i] - self.start_object_locations[j]) for j in range(self.num_objects)] for i in range(self.num_objects)  ])

        self.master_gcs_program = GraphOfConvexSets()
        self.name_to_vertex = dict()
        self.name_to_edge = dict()
        self.possible_block_to_edge_list = dict() # type: T.Dict[int, T.List[GraphOfConvexSets.Edge]]
        for i in range(2 * self.num_objects):
            self.possible_block_to_edge_list[i] = []

        self.build_master_program()

    def find_cost_change_due_to_swap(self, swap_index, new_order):
        # cost of moving i-th object
        solution = self.solve_for_path_from_to(new_order[swap_index], new_order)
        if not solution.is_success():
            return False, []
        optimal_cost_of_1 = solution.get_optimal_cost()

        # cost of moving i+1-th object
        solution = self.solve_for_path_from_to(new_order[swap_index+1], new_order)
        if not solution.is_success():
            return False, []
        optimal_cost_of_2 = solution.get_optimal_cost()

        return True, [optimal_cost_of_1, optimal_cost_of_2]


    def solve_from_feasible_solution(self, feasible_order: T.List[int]):
        x = timeit()
        assert type(feasible_order) == list
        n = self.num_objects
        current_order = feasible_order.copy()

        original_order_cost = 0
        cost_of_moving_object_i_at_current_order = [0] * n

        # TODO: PARALLELIZE ME
        for i in range(n):
            solution = self.solve_for_path_from_to(current_order[i], current_order)
            assert solution.is_success(), "provided feasible order was not feasible for " + str(current_order[i]) + " at order " + str(i) 
            cost_of_moving_object_i_at_current_order[i] = solution.get_optimal_cost()
        
        original_order_cost += sum(cost_of_moving_object_i_at_current_order)
        original_order_cost += self.arm_start_to_si[ current_order[0] ]
        for i in range(n-1):
            original_order_cost += self.ti_to_sj[ current_order[i], current_order[i+1]  ]
        original_order_cost += self.ti_to_arm_target[ current_order[i+1] ]

        # there are n-1 promising swaps
        # a swap is not promising if it has been evaluated now / at previous step
        # no neighbor of that swap was evaluated at last iteration

        # at start, each swap is promising
        promising_swaps = [1] * (n-1)

        num_iterations = 0
        # continue while there are promising swaps
        while sum(promising_swaps) > 0:
            print()
            num_iterations += 1
            improving_swaps = []
            
            # for each swap:
            for i in range(n-1):
                # if the swap is promising
                if promising_swaps[i] == 1:
                    promising_swaps[i] = 0
                    # i need to revaluate two costs: moving i-th object, moving i+1 object
                    new_order = current_order[:i] + [current_order[i+1]] + [current_order[i]] + current_order[i+2:]
                
                    # calculate cost delta due to this swap
                    feasible, prog_costs = self.find_cost_change_due_to_swap(i, new_order)

                    if not feasible:
                        continue

                    # calculate improvement of this swap over the current order
                    cost_delta = 0
                    cost_delta += cost_of_moving_object_i_at_current_order[i]
                    cost_delta += cost_of_moving_object_i_at_current_order[i+1]
                    cost_delta -= (prog_costs[0] + prog_costs[1])

                    # cost of moving from previous location to i
                    if i == 0:
                        cost_delta += self.arm_start_to_si[ current_order[i] ]
                        cost_delta -= self.arm_start_to_si[ new_order[i] ]
                    else:
                        cost_delta += self.ti_to_sj[ current_order[i-1], current_order[i]  ]
                        cost_delta -= self.ti_to_sj[ new_order[i-1], new_order[i]  ]

                    # cost of moving from i to i+1
                    cost_delta += self.ti_to_sj[ current_order[i], current_order[i+1]  ]
                    cost_delta -= self.ti_to_sj[ new_order[i], new_order[i+1]  ]

                    # cost of moving from i+1 to next location
                    if i == self.num_objects-2:
                        cost_delta += self.ti_to_arm_target[ current_order[i+1] ]
                        cost_delta -= self.ti_to_arm_target[ new_order[i+1] ]
                    else:
                        cost_delta += self.ti_to_sj[ current_order[i+1], current_order[i+2]  ]
                        cost_delta -= self.ti_to_sj[ new_order[i+1], new_order[i+2]  ]

                    # if swap i is an improvement -- add it to the list of improving_swaps
                    if cost_delta > 0:
                        # improvement observed, yay!
                        improving_swaps.append( [i, cost_delta] + prog_costs )

            # make a list of accepted swaps
            # TODO: heuristics for accepting swaps?

            accepted_swap_indecis = []
            accepted_swaps = []

            # sort by best ones; first is best
            # accept as many swaps as you can; no two consecutive swaps are allowed
            improving_swaps.sort(key = lambda x: -x[1])
            for swap_array in improving_swaps:
                # no two consecutive swaps are allowed
                if swap_array[0]-1 not in accepted_swap_indecis and swap_array[0]+1 not in accepted_swap_indecis:
                    accepted_swap_indecis.append(swap_array[0])
                    accepted_swaps.append(swap_array)


            for swap_array in accepted_swaps:
                i = swap_array[0]
                i_cost = swap_array[2]
                i_1_cost = swap_array[3]

                # swap
                current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
                # update costs
                cost_of_moving_object_i_at_current_order[i] = i_cost
                cost_of_moving_object_i_at_current_order[i+1] = i_1_cost
                # udpate promising swaps:
                if i > 0:
                    promising_swaps[i-1] = 1
                if i < n-2:
                    promising_swaps[i+1] = 1
        
        

        new_order_cost = 0
        new_order_cost += sum(cost_of_moving_object_i_at_current_order)
        new_order_cost += self.arm_start_to_si[ current_order[0] ]
        for i in range(n-1):
            new_order_cost += self.ti_to_sj[ current_order[i], current_order[i+1]  ]
        new_order_cost += self.ti_to_arm_target[ current_order[i+1] ]

        x.dt("Iterative search took ")
        INFO("Number of iterations: ", num_iterations)
        INFO("Number of GCS solves: ", self.num_gcs_solves)
        WARN("Original order: ", feasible_order)
        YAY("New order: ", current_order)
        WARN("Cost of original order", original_order_cost)
        YAY("Cost of new order: ", new_order_cost )
        YAY("-----------")
        YAY("Cost improvement: ", original_order_cost - new_order_cost)
        return current_order


        
    def solve_for_path_from_to(self, object_index: int, order:npt.NDArray):
        prog = self.master_gcs_program.copy()
        self.num_gcs_solves += 1
        
        prog.ClearAllPhiConstraints()
        # don't allow going through the following objects
        offset = 0
        for i in range(self.num_objects):
            if order[i] == object_index:
                offset = self.num_objects
            else:
                for edge in self.possible_block_to_edge_list[ order[i] + offset ]:
                    edge.AddPhiConstraint(False)

        start_vertex = self.name_to_vertex["s" + str(object_index)].id()
        target_vertex = self.name_to_vertex["t" + str(object_index)].id()

        options = opt.GraphOfConvexSetsOptions()
        options.convex_relaxation = True
        options.preprocessing = False  # TODO Do I need to deal with this?
        options.max_rounded_paths = 10
        options.rounding_seed = 1

        solution = self.master_gcs_program.SolveShortestPath(start_vertex, target_vertex, options)
        return solution


    def build_master_program(self):
        # add all vertices
        for v in self.tessellation_graph.vertices.values():
            vertex = self.master_gcs_program.AddVertex(v.get_hpolyhedron(), v.name)
            self.name_to_vertex[v.name] = vertex

        # add all edges between sets
        for e in self.tessellation_graph.edges.values():
            edge_name = e.left.name + " -> " + e.right.name
            edge = self.master_gcs_program.AddEdge(self.name_to_vertex[e.left.name], self.name_to_vertex[e.right.name], edge_name)
            self.name_to_edge[edge_name] = edge
            # if right vertex of this edge represents some occupied objects
            for object_index in self.tessellation_graph.vertices[e.right.name].aligned_set.objects:
                self.possible_block_to_edge_list[object_index].append(edge)
            
            # add "right in set left set's set"
            self.add_common_set_at_transition_constraint(edge, self.name_to_vertex[e.left.name].set())
            
            # add an L2-cost over an edge
            self.add_l2_norm(edge)

        # get start-target sets for each object, add them as vertices
        for i in range(self.num_objects):
            set_vertex_name = self.tessellation_graph.find_graph_vertex_that_has_point( self.start_object_locations[i] )
            start_name = "s" + str(i)
            edge_name = start_name + " -> " + set_vertex_name
            start_vertex = self.master_gcs_program.AddVertex(Point(np.array(self.start_object_locations[i])), start_name)
            self.name_to_vertex[start_name] = start_vertex

            edge = self.master_gcs_program.AddEdge(start_vertex, self.name_to_vertex[set_vertex_name], edge_name)
            self.name_to_edge[edge_name] = edge
            # add l2-norm
            self.add_l2_norm(edge)
            
        for i in range(self.num_objects):
            set_vertex_name = self.tessellation_graph.find_graph_vertex_that_has_point( self.target_object_locations[i] )
            target_name = "t" + str(i)
            edge_name = set_vertex_name + " -> " + target_name
            target_vertex = self.master_gcs_program.AddVertex(Point(np.array(self.target_object_locations[i])), target_name)
            self.name_to_vertex[target_name] = target_vertex

            edge = self.master_gcs_program.AddEdge(self.name_to_vertex[set_vertex_name], target_vertex, edge_name)
            self.name_to_edge[edge_name] = edge
            # add l2-norm
            self.add_l2_norm(edge)
            
    def add_common_set_at_transition_constraint(self, edge: GraphOfConvexSets.Edge, left_vertex_set: HPolyhedron):
        A = left_vertex_set.A()
        lb = -np.ones(left_vertex_set.b().size) * 1000
        ub = left_vertex_set.b()
        set_con = LinearConstraint(A, lb, ub)
        edge.AddConstraint(Binding[LinearConstraint](set_con, edge.xv()))


    def add_l2_norm(self, edge: GraphOfConvexSets.Edge):
        n = 2
        A, b = np.hstack((np.eye(n), -np.eye(n))), np.zeros(n)
        cost = L2NormCost(A, b)
        edge.AddCost(Binding[L2NormCost](cost, np.append(edge.xv(), edge.xu())))

    def find_path_to_target(
        self,
        edges: T.List[GraphOfConvexSets.Edge],
        start: GraphOfConvexSets.Vertex,
        target: GraphOfConvexSets.Vertex,
    ) -> T.List[GraphOfConvexSets.Vertex]:
        """Given a set of active edges, find a path from start to target"""

        edges_out = [e for e in edges if e.u() == start]
        current_edge = edges_out[0]
        v = edges_out[0].v()

        target_reached = v == target

        if target_reached:
            return [start] + [v], [current_edge]
        else:
            v, e = self.find_path_to_target(edges, v, target)
            return [start] + v, [current_edge] + e

    def get_solution_path(self, solution, start_name, taget_name) -> T.Tuple[T.List[str], npt.NDArray]:
        """Given a solved GCS problem, and assuming it's tight, find a path from start to target"""
        # find edges with non-zero flow
        flow_variables = [e.phi() for e in self.master_gcs_program.Edges()]
        flow_results = [solution.GetSolution(p) for p in flow_variables]
        not_tight = np.any(
            np.logical_and(0.05 < np.array(flow_results), np.array(flow_results) < 0.95)
        )

        active_edges = [edge for edge, flow in zip(self.master_gcs_program.Edges(), flow_results) if flow > 0.0]

        target_vertex = self.name_to_vertex[taget_name]

        # using these edges, find the path from start to target
        path, _ = self.find_path_to_target(active_edges, self.name_to_vertex[start_name], target_vertex)
        vertex_names = [v.name() for v in path]
        vertex_values = np.vstack([solution.GetSolution(v.x()) for v in path])
        return vertex_names, vertex_values

    def display_graph(self, solution = None, graph_name="temp") -> None:
        """Visually inspect the graph. If solution acquired -- also displays the solution."""
        if solution is None or not solution.is_success():
            graphviz = self.master_gcs_program.GetGraphvizString()
        else:
            graphviz = self.master_gcs_program.GetGraphvizString(solution, True, precision=2)
        data = pydot.graph_from_dot_data(graphviz)[0]  # type: ignore
        data.write_png(graph_name + ".png")
        data.write_svg(graph_name + ".svg")
        

def simple_small_test():
    tess, start_locs, target_locs = make_a_test_with_objects_and_obstacles()
    tess_graph = GraphOfAdjacentAlignedSets(tess)
    num_objects = 2
    arm_start = np.array([1,7])
    arm_target = np.array([3,3])
    prog = DescentOverPermutahedron(num_objects, arm_start, arm_target, start_locs, target_locs, tess_graph)
    prog.solve_from_feasible_solution( [1, 0] )

if __name__ == "__main__":

    lb, ub = 0,30

    bounding_box = AlignedSet(a=ub, b = lb, l = lb, r = ub, set_type = FREE )
    obstacle_sets = []

    n = 10
    np.random.seed(4)
    start_object_locations = [tuple(np.random.uniform( lb+1, ub-1, 2)) for i in range(n)]
    target_object_locations = [tuple(np.random.uniform( lb+1, ub-1, 2)) for i in range(n)]

    start_arm_position = np.array([0,0])
    target_arm_position = np.array([0,0])

    program_options = ProgramOptionsForGCSTSP()
    
    program_options.add_tsp_edge_costs = True
    program_options.convex_relaxation_for_gcs_edges = True

    program_options.add_L2_norm_cost = False
    program_options.solve_for_feasibility = True

    full_prog = GraphTSPGCSProgram.construct_from_positions(bounding_box, obstacle_sets, start_object_locations, target_object_locations, start_arm_position, target_arm_position, program_options)
    full_prog.solve()

    feasible_order = full_prog.extract_order()
    print(feasible_order)


    # descent_prog = DescentOverPermutahedron(n, start_arm_position, target_arm_position, start_object_locations, target_object_locations, full_prog.tessellation_graph)
    # descent_prog.solve_from_feasible_solution(feasible_order)
        
