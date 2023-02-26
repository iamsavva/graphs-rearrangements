import typing as T

import numpy as np
import numpy.typing as npt

from pydrake.solvers import (
    MathematicalProgram,
    Solve,
)  # pylint: disable=import-error, no-name-in-module
from pydrake.math import le, eq  # pylint: disable=import-error, no-name-in-module

from util import timeit

from vertex import Vertex, VertexTSPprogram, VertexAlignedSet, VertexTSP
from edge import Edge, EdgeMotionPlanningProgam
from graph_tsp_gcs import ProgramOptionsForGCSTSP
from axis_aligned_set import AlignedSet, Box
from axis_aligned_set_tesselation import (
    AxisAlignedSetTessellation,
    make_a_test_with_objects_and_obstacles,
)
from axis_aligned_graph import GraphOfAdjacentAlignedSets


class MotionPlanningProgram:
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
        prog: MathematicalProgram,  # overall MICP
        vertices: T.Dict[str, Vertex],  # vertices from overall MICP
        edges: T.Dict[str, Edge],  # edges from overall MICP
        start_vertex: VertexTSPprogram,
        target_vertex: VertexTSPprogram,
        tessellation_graph: GraphOfAdjacentAlignedSets,  # probably a graph here
        options: ProgramOptionsForGCSTSP,
    ) -> None:

        self.tessellation_graph = tessellation_graph

        # start-target related data structures
        self.start_object_index = (
            start_vertex.possible_object_index
        )  # this is in the vertex
        self.target_object_index = target_vertex.possible_object_index

        self.start_tsp_vertex_name = start_vertex.name
        self.target_tsp_vertex_name = target_vertex.name

        self.start_position = start_vertex.block_position
        self.target_position = target_vertex.block_position

        self.start_tessellation_vertex_name = (
            self.tessellation_graph.find_graph_vertex_that_has_point(
                self.start_position
            )
        )
        self.target_tessellation_vertex_name = (
            self.tessellation_graph.find_graph_vertex_that_has_point(
                self.target_position
            )
        )

        self.options = options

        # suffix for naming
        self.mp_vertex_suffix = "_mp" + str(self.start_object_index)  # type: str

        self.prog = prog  # type: MathematicalProgram

        # vertices of the entire program
        self.all_vertices = vertices
        self.all_edges = edges

        # specifically motion-planning-relevant vertices -- duplicating for access
        self.vertices = dict()  # type: T.Dict[str, VertexAlignedSet]
        self.edges = dict()  # type: T.Dict[str, EdgeMotionPlanningProgam]

        self.vertices[self.start_tsp_vertex_name] = self.all_vertices[
            self.start_tsp_vertex_name
        ]
        self.vertices[self.target_tsp_vertex_name] = self.all_vertices[
            self.target_tsp_vertex_name
        ]

        self.add_mp_vertices_and_edges()
        self.add_mp_variables_to_prog()
        self.add_mp_constraints_to_prog()
        self.add_mp_costs_to_prog()

    def mp_name(self, name):
        return name + self.mp_vertex_suffix

    def add_vertex(self, vertex_aligned_set: VertexAlignedSet) -> None:
        """
        Add a new vertex to both full vertex set and local vertex set.
        Note that this implementation differs from TSP -- bc obstacles.
        """
        # modify the name
        name = self.mp_name(vertex_aligned_set.name)
        aligned_set = vertex_aligned_set.aligned_set
        aligned_set.name = name

        assert name not in self.vertices, "Vertex with name " + name + " already exists"
        assert name not in self.all_vertices, (
            "Vertex with name " + name + " already exists in og graph"
        )

        self.all_vertices[name] = VertexAlignedSet(name, aligned_set)
        self.vertices[name] = self.all_vertices[name]

    def add_edge(self, left_name: str, right_name: str) -> None:
        """
        Add a new edge to both full edge set and local edge set.
        """
        edge_name = left_name + "_" + right_name
        assert edge_name not in self.edges, (
            "Edge " + edge_name + " already in new edges"
        )
        assert edge_name not in self.all_edges, (
            "Edge " + edge_name + " already in og edges"
        )
        self.all_edges[edge_name] = EdgeMotionPlanningProgam(
            self.all_vertices[left_name], self.all_vertices[right_name], edge_name
        )
        self.edges[edge_name] = self.all_edges[edge_name]
        self.all_vertices[left_name].add_edge_out(edge_name)
        self.all_vertices[right_name].add_edge_in(edge_name)

    def add_mp_vertices_and_edges(self) -> None:
        """
        Graph structure: add motion planning vertices and edges.
        """
        ############################
        # tsp start/target should already be in vertices
        assert self.start_tsp_vertex_name in self.vertices
        assert self.target_tsp_vertex_name in self.vertices

        # add mp vertices
        for aligned_set_vertex in self.tessellation_graph.vertices.values():
            self.add_vertex(aligned_set_vertex)

        ############################
        # add all edges
        # add edge from between tsp portion and mp portion
        self.add_edge(
            self.start_tsp_vertex_name,
            self.mp_name(self.start_tessellation_vertex_name),
        )
        self.add_edge(
            self.mp_name(self.target_tessellation_vertex_name),
            self.target_tsp_vertex_name,
        )

        # add all edges within the mp portion
        for edge in self.tessellation_graph.edges.values():
            self.add_edge(self.mp_name(edge.left.name), self.mp_name(edge.right.name))

    def add_mp_variables_to_prog(self) -> None:
        """
        Program variables -- add variables on the edges -- flows and position.
        """
        ###################################
        # add edge variables
        for e in self.edges.values():
            # add flow variable
            if self.options.convex_relaxation_for_gcs_edges:
                # cotninuous variable, flow between 0 and 1
                e.set_phi(self.prog.NewContinuousVariables(1, "phi_" + e.name)[0])
                self.prog.AddLinearConstraint(e.phi, 0.0, 1.0)
            else:
                e.set_phi(self.prog.NewBinaryVariables(1, "phi_" + e.name)[0])

            # if we are solving for feasibility -- we are solving a graph problem. no need for GCS type stuff.
            if not self.options.solve_for_feasibility:
                # if the edge is not from start tsp or end at target tsp
                left_not_tsp_vertex = e.left.name != self.start_tsp_vertex_name
                right_not_tsp_vertex = e.right.name != self.target_tsp_vertex_name
                if left_not_tsp_vertex and right_not_tsp_vertex:
                    e.set_l_pos(self.prog.NewContinuousVariables(2, "l_pos_" + e.name))
                    e.set_r_pos(self.prog.NewContinuousVariables(2, "r_pos_" + e.name))

    def add_mp_constraints_to_prog(self) -> None:
        """
        Motion planning constraints -- good old motion planning GCS, with some tricks for turning
        obstacles on and off.
        """
        ###################################
        # PER VERTEX
        # sum over edges in of l_pos = sum over edges out of r_pos
        # flow in = flow_out
        for v in self.vertices.values():
            ##############################
            # pos_out = pos_in constraints
            if not self.options.solve_for_feasibility:
                # it's a start node
                if v.name not in (
                    self.start_tsp_vertex_name,
                    self.target_tsp_vertex_name,
                    self.mp_name(self.target_tessellation_vertex_name),
                    self.mp_name(self.start_tessellation_vertex_name),
                ):
                    pos_out = sum(
                        [
                            self.edges[e].l_pos
                            for e in v.edges_out
                            if self.edges[e].l_pos is not None
                        ]
                    )
                    pos_in = sum(
                        [
                            self.edges[e].r_pos
                            for e in v.edges_in
                            if self.edges[e].r_pos is not None
                        ]
                    )
                    self.prog.AddLinearConstraint(eq(pos_out, pos_in))

            ##############################
            # flow in = flow_out constraints
            # NOTE: these constraints are neceesary here as MP edges are added after TSP edges
            if v.name == self.start_tsp_vertex_name:
                flow_out = sum([self.edges[e].phi for e in v.edges_out])
                self.prog.AddLinearConstraint(flow_out == 1)
            elif v.name == self.target_tsp_vertex_name:
                flow_in = sum([self.edges[e].phi for e in v.edges_in])
                self.prog.AddLinearConstraint(flow_in == 1)
            else:
                # TODO: why was bothered by this??
                # if v.name == self.mp_name(self.start_tessellation_vertex_name):
                    # print("1")
                flow_in = sum([self.edges[e].phi for e in v.edges_in])
                flow_out = sum([self.edges[e].phi for e in v.edges_out])
                self.prog.AddLinearConstraint(flow_in == flow_out)
                # NOTE: having all variables be bounded makes optimization faster
                self.prog.AddLinearConstraint(flow_in <= 1)
                self.prog.AddLinearConstraint(flow_out <= 1)

        ###################################
        # PER EDGE
        # (flow, l_pos) are in perspective set of left set
        # (flow, r_pos) are in perspective set of left set and right set (at intersection)
        # (flow, visitation[i]) for s_i and t_i must belong to certain sets -- on/off obstacles
        for e in self.edges.values():
            # for each motion planning edge
            if not self.options.solve_for_feasibility:
                left_not_tsp_vertex = e.left.name != self.start_tsp_vertex_name
                right_not_tsp_vertex = e.right.name != self.target_tsp_vertex_name
                if left_not_tsp_vertex and right_not_tsp_vertex:
                    lA, lb = e.left.aligned_set.get_perspective_hpolyhedron_matrices()
                    rA, rb = e.right.aligned_set.get_perspective_hpolyhedron_matrices()
                    # left is in the set that corresponds to left
                    self.prog.AddLinearConstraint(
                        le(lA @ np.append(e.l_pos, e.phi), lb)
                    )
                    # right is in the set that corresponds to left and right
                    self.prog.AddLinearConstraint(
                        le(lA @ np.append(e.r_pos, e.phi), lb)
                    )
                    self.prog.AddLinearConstraint(
                        le(rA @ np.append(e.r_pos, e.phi), rb)
                    )
            # NOTE: i am not adding these constraints on edge into self.target_tsp
            # such constraint is redundant because there is unique edge into target_tsp

            ###################################
            # turning obstacles on and off: don't allow to enter set if it's an obstacle
            # flow and visitaiton of that obstacle must belong to a particular set
            # if this is part of obstacles
            # for each object it's part of
            if type(e.right) == VertexAlignedSet:
                for object_index in e.right.aligned_set.objects:
                    # don't add obstacles that correspond to moving yourself
                    if object_index == self.start_object_index:
                        continue
                    # don't make our start/target mp vertices into an obstacle!

                    # this should not be relevant?
                    # if e.right.name == self.start_mp and obst_type == "s":
                    #     continue
                    # if e.right.name == self.target_mp and obst_type == "t":
                    #     continue

                    # [visitation of i , flow into i] must belong to a particular set given by A, b
                    x = np.array(
                        [
                            self.vertices[self.start_tsp_vertex_name].v[object_index],
                            e.phi,
                        ]
                    )
                    A = np.array([[-1, 0], [0, -1], [1, 1]])
                    b = np.array([0, 0, 1])  # TODO: may be the other one
                    # if obst_type == "s":
                    #     A = np.array([[1, 0], [0, -1], [-1, 1]])
                    #     b = np.array([1, 0, 0])
                    # elif obst_type == "t":
                    #     A = np.array([[-1, 0], [0, -1], [1, 1]])
                    #     b = np.array([0, 0, 1])
                    # else:
                    #     raise Exception("non start-target obstacle?")
                    self.prog.AddLinearConstraint(le(A @ x, b))

    def add_mp_costs_to_prog(self) -> None:
        """
        Motion planning costs: L2 norm over travelled distance, defined as a SOC constraint.
        """
        ###################################
        # PER EDGE
        # L2 norm over travelled distance, defined as a SOC constraint
        # TODO: it is annoying that there are a bunch of ~random zero-length non-zero edges
        #       they don't do anything, but still annoying.
        if self.options.add_L2_norm_cost:
            for e in self.edges.values():
                left_not_tsp_vertex = e.left.name != self.start_tsp_vertex_name
                right_not_tsp_vertex = e.right.name != self.target_tsp_vertex_name
                if left_not_tsp_vertex and right_not_tsp_vertex:
                    # ||r_pos - l_pos||_2

                    A = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
                    b = np.array([0, 0])

                    self.prog.AddL2NormCostUsingConicConstraint(
                        A, b, np.append(e.l_pos, e.r_pos)
                    )
                
                if e.left.name == self.mp_name(self.start_tessellation_vertex_name) and "tsp" not in e.right.name:
                    A = np.array([[1, 0], [0, 1]])
                    b = -self.start_position
                    self.prog.AddL2NormCostUsingConicConstraint(A, b, e.l_pos)
                
                if e.right.name == self.mp_name(self.target_tessellation_vertex_name) and "tsp" not in e.left.name:
                    A = np.array([[1, 0], [0, 1]])
                    b = -self.target_position
                    self.prog.AddL2NormCostUsingConicConstraint(A, b, e.r_pos)


def test():
    (
        tessellation,
        object_start_locs,
        object_target_locs,
    ) = make_a_test_with_objects_and_obstacles()
    tessellation_graph = GraphOfAdjacentAlignedSets(tessellation)
    num_possible_objects = 4

    prog = MathematicalProgram()
    vertices = dict()
    edges = dict()

    print(object_start_locs[0])
    print(object_target_locs[0])

    # make vertices
    start_vertex = VertexTSPprogram("start_1_tsp", np.array(object_start_locs[0]), 0)
    start_vertex.set_v(np.array([1, 1, 0, 0]))

    target_vertex = VertexTSPprogram("target_1_tsp", np.array(object_target_locs[0]), 2)

    # add vertices
    vertices[start_vertex.name] = start_vertex
    vertices[target_vertex.name] = target_vertex
    # add initial visitation state

    # prog.AddLinearConstraint( eq(start_vertex.v, np.array([1,1,0,0]) ) )

    # set up the solve options
    options = ProgramOptionsForGCSTSP()
    options.add_L2_norm_cost = False
    options.convex_relaxation_for_gcs_edges = True
    options.solve_for_feasibility = True

    # populate the program with motion planning stuff
    MotionPlanningProgram(
        prog, vertices, edges, start_vertex, target_vertex, tessellation_graph, options
    )

    x = timeit()
    solution = Solve(prog)
    x.dt()
    infeasible_constraints = solution.GetInfeasibleConstraints(prog)
    for infeas in infeasible_constraints:
        print(infeas)

    print(solution.is_success())
    print(solution.get_optimal_cost())

    flows = [edge.name for edge in edges.values() if solution.GetSolution(edge.phi) > 0]

    # for edge in edges.values():
    #     print(edge.name, solution.GetSolution(edge.phi))

    tessellation_graph.plot_the_tessellation_graph()


if __name__ == "__main__":
    test()
