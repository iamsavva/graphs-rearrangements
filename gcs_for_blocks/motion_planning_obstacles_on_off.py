import typing as T

import numpy as np
import numpy.typing as npt

from pydrake.solvers import (
    MathematicalProgram,
    L1NormCost,
)  # pylint: disable=import-error, no-name-in-module
from pydrake.math import le, eq  # pylint: disable=import-error, no-name-in-module

# from .util import timeit, INFO, WARN, ERROR, YAY
# from .axis_aligned_set_tesselation import (
#     AlignedSet,
# )
# from .tsp_vertex_edge import Vertex, Edge

from graph import Vertex, Edge
from graph_tsp_gcs import VertexTSPprogram, EdgeTSPprogram
from graph_tsp_gcs import ProgramOptionsForGCSTSP
from axis_aligned_set import AlignedSet, Box
from axis_aligned_set_tesselation import AxisAlignedSetTessellation


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
        set_tessellation: AxisAlignedSetTessellation, # probably a graph here
        options: ProgramOptionsForGCSTSP
    ) -> None:
        

        
        self.start_object_index = start_vertex.possible_object_index # this is in the vertex
        self.target_object_index = target_vertex.possible_object_index

        self.start_tsp_vertex_name = start_vertex.name
        self.target_tsp_vertex_name = target_vertex.name

        self.start_set_name = "" # how do you find? start_vertex.position
        self.target_set_name = "" # how do you find it? target_vertex.position

        




        
        self.options = options




        # self.num_blocks = len(start_block_pos)  # type: int
        self.moving_block_index = moving_block_index  # type: int
        self.start_block_pos = [
            np.array(x) for x in start_block_pos
        ]  # type: T.List[npt.NDArray]
        self.target_block_pos = [
            np.array(x) for x in target_block_pos
        ]  # type: T.List[npt.NDArray]
        
        # start / target node names
        smbi = str(self.moving_block_index)  # type: str
        self.start_tsp = "s" + smbi + "_tsp"  # type: str
        self.target_tsp = "t" + smbi + "_tsp"  # type: str
        self.start_mp = self.mp_name(obstacle_to_set["s" + smbi])  # type: str
        self.target_mp = self.mp_name(obstacle_to_set["t" + smbi])  # type: str


        # rename the sets in the convex set tesselation
        self.convex_set_tesselation = dict()
        for name in convex_set_tesselation:
            new_name = self.mp_name(name)
            self.convex_set_tesselation[new_name] = convex_set_tesselation[name].copy()
            self.convex_set_tesselation[new_name].name = new_name
        assert len(target_block_pos) == self.num_blocks

        self.prog = prog  # type: MathematicalProgram
        self.convex_relaxation = convex_relaxation  # type: bool
        self.share_edge_tol = share_edge_tol  # type: float
        # vertecis of the entire program
        self.all_vertices = all_vertices
        self.all_edges = all_edges
        # specifically motion-planning-relevant vertices
        self.vertices = dict()  # type: T.Dict[str, Vertex]
        self.edges = dict()  # type: T.Dict[str, Edge]
        self.vertices[self.start_tsp] = self.all_vertices[self.start_tsp]
        self.vertices[self.target_tsp] = self.all_vertices[self.target_tsp]

        self.add_mp_vertices_and_edges()
        self.add_mp_variables_to_prog()
        self.add_mp_constraints_to_prog()

        # CHANGE ME
        self.add_mp_costs_to_prog()

    def mp_name(self, name):
        return name + "_mp" + str(self.moving_block_index)

    def add_vertex(self, aligned_set: AlignedSet) -> None:
        """
        Add a new vertex to both full vertex set and local vertex set.
        Note that this implementation differs from TSP -- bc obstacles.
        """
        name = aligned_set.name
        assert name not in self.vertices, "Vertex with name " + name + " already exists"
        assert name not in self.all_vertices, (
            "Vertex with name " + name + " already exists in og"
        )
        self.all_vertices[name] = Vertex(name, block_index=self.moving_block_index)
        if len(aligned_set.obstacles) > 0:
            self.all_vertices[name].set_obstacles(aligned_set.obstacles)
        self.vertices[name] = self.all_vertices[name]

    def add_edge(self, left_name: str, right_name: str) -> None:
        """
        Add a new edge to both full edge set and local edge set.
        """
        # the kind of edge i add depends heavily

        edge_name = left_name + "_" + right_name
        assert edge_name not in self.edges, (
            "Edge " + edge_name + " already exists in new edges"
        )
        assert edge_name not in self.all_edges, (
            "Edge " + edge_name + " already exists in og edges"
        )
        self.all_edges[edge_name] = Edge(
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
        assert self.start_tsp in self.vertices
        assert self.target_tsp in self.vertices

        # add mp vertices
        for aligned_set in self.convex_set_tesselation.values():
            self.add_vertex(aligned_set)

        ############################
        # add all edges
        # add edge from between tsp portion and mp portion
        self.add_edge(self.start_tsp, self.start_mp)
        self.add_edge(self.target_mp, self.target_tsp)
        # add all edges within the mp portion
        for set1 in self.convex_set_tesselation.values():
            for set2 in self.convex_set_tesselation.values():
                # no repeats
                if set1 != set2 and set1.share_edge(set2, self.share_edge_tol):
                    # add edge between set1 and set2
                    self.add_edge(set1.name, set2.name)

    def add_mp_variables_to_prog(self) -> None:
        """
        Program variables -- add variables on the edges -- flows and position.
        """
        ###################################
        # add edge variables
        for e in self.edges.values():
            # add flow variable
            if self.convex_relaxation:
                # cotninuous variable, flow between 0 and 1
                e.set_phi(self.prog.NewContinuousVariables(1, "phi_" + e.name)[0])
                self.prog.AddLinearConstraint(e.phi, 0.0, 1.0)
            else:
                e.set_phi(self.prog.NewBinaryVariables(1, "phi_" + e.name)[0])

            # if the edge is not from start
            if e.left.name != self.start_tsp:
                e.set_left_pos(
                    self.prog.NewContinuousVariables(2, "left_pos_" + e.name)
                )
                e.set_right_pos(
                    self.prog.NewContinuousVariables(2, "right_pos_" + e.name)
                )

    def add_mp_constraints_to_prog(self) -> None:
        """
        Motion planning constraints -- good old motion planning GCS, with some tricks for turning
        obstacles on and off.
        """
        ###################################
        # PER VERTEX
        # sum over edges in of left_pos = sum over edges out of right_pos
        # flow in = flow_out
        for v in self.vertices.values():
            ##############################
            # pos_out = pos_in constraints
            # it's a start node
            if v.name == self.start_tsp:
                continue
            # it's a target node
            elif v.name == self.target_tsp:
                pos_in = sum([self.edges[e].right_pos for e in v.edges_in])
                # sum pos in is the target-pos
                block_target_pos = self.target_block_pos[self.moving_block_index]
                self.prog.AddLinearConstraint(eq(pos_in, block_target_pos))
            # it's a start-set node
            elif v.name == self.start_mp:
                pos_out = sum([self.edges[e].left_pos for e in v.edges_out])
                # sum pos out is the start-pos
                block_start_pos = self.start_block_pos[self.moving_block_index]
                self.prog.AddLinearConstraint(eq(pos_out, block_start_pos))
            # it's any other old node
            else:
                pos_out = sum([self.edges[e].left_pos for e in v.edges_out])
                pos_in = sum([self.edges[e].right_pos for e in v.edges_in])
                # sum of y equals sum of z
                self.prog.AddLinearConstraint(eq(pos_out, pos_in))

            ##############################
            # flow in = flow_out constraints
            if v.name == self.start_tsp:
                flow_out = sum([self.edges[e].phi for e in v.edges_out])
                self.prog.AddLinearConstraint(flow_out == 1)
            elif v.name == self.target_tsp:
                flow_in = sum([self.edges[e].phi for e in v.edges_in])
                self.prog.AddLinearConstraint(flow_in == 1)
            else:
                flow_in = sum([self.edges[e].phi for e in v.edges_in])
                flow_out = sum([self.edges[e].phi for e in v.edges_out])
                self.prog.AddLinearConstraint(flow_in == flow_out)
                self.prog.AddLinearConstraint(flow_in <= 1)
                self.prog.AddLinearConstraint(flow_out <= 1)

        ###################################
        # PER EDGE
        # (flow, left_pos) are in perspective set of left set
        # (flow, right_pos) are in perspective set of left set and right set (at intersection)
        # (flow, visitation[i]) for s_i and t_i must belong to certain sets -- on/off obstacles
        for e in self.edges.values():
            # for each motion planning edge
            if e.left.name != self.start_tsp and e.right.name != self.target_tsp:
                left_aligned_set = self.convex_set_tesselation[e.left.name]
                lA, lb = left_aligned_set.get_perspective_hpolyhedron()
                right_aligned_set = self.convex_set_tesselation[e.right.name]
                rA, rb = right_aligned_set.get_perspective_hpolyhedron()
                # left is in the set that corresponds to left
                self.prog.AddLinearConstraint(le(lA @ np.append(e.left_pos, e.phi), lb))
                # right is in the set that corresponds to left and right
                self.prog.AddLinearConstraint(
                    le(lA @ np.append(e.right_pos, e.phi), lb)
                )
                self.prog.AddLinearConstraint(
                    le(rA @ np.append(e.right_pos, e.phi), rb)
                )
            # NOTE: i am not adding these constraints on edge into self.target_tsp
            # such constraint is redundant because there is unique edge into target_tsp

            ###################################
            # turning obstacles on and off: don't allow to enter set if it's an obstacle
            # flow and visitaiton of that obstacle must belong to a particular set
            # if this is part of obstacles
            if e.right.obstacles is not None:
                # for each obstacle it's part of
                for (obst_type, obstacle_num) in e.right.obstacles:
                    # don't add obstacles that correspond to moving yourself
                    if obstacle_num == self.moving_block_index:
                        continue
                    # don't make our start/target mp vertices into an obstacle!
                    if e.right.name == self.start_mp and obst_type == "s":
                        continue
                    if e.right.name == self.target_mp and obst_type == "t":
                        continue
                    # [visitation of i , flow into i] must belong to a particular set given by A, b
                    x = np.array([self.vertices[self.start_tsp].v[obstacle_num], e.phi])
                    if obst_type == "s":
                        A = np.array([[1, 0], [0, -1], [-1, 1]])
                        b = np.array([1, 0, 0])
                    elif obst_type == "t":
                        A = np.array([[-1, 0], [0, -1], [1, 1]])
                        b = np.array([0, 0, 1])
                    else:
                        raise Exception("non start-target obstacle?")
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
                if e.left.name != self.start_tsp:
                    # ||right_pos - left_pos||_2
                    A = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
                    b = np.array([0, 0])

                    self.prog.AddL2NormCostUsingConicConstraint(
                        A, b, np.append(e.left_pos, e.right_pos)
                    )
