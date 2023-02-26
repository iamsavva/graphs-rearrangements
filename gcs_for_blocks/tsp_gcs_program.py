import typing as T

import numpy as np
import numpy.typing as npt
from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module
    MathematicalProgram,
    Solve,
)
from pydrake.math import le, eq  # pylint: disable=import-error, no-name-in-module

# from graph import Vertex, Edge, Graph
from axis_aligned_set import AlignedSet, Box
from axis_aligned_set_tesselation import AxisAlignedSetTessellation

# from graph import Vertex, Edge
from vertex import VertexTSP, VertexTSPprogram
from edge import EdgeTSP, EdgeTSPprogram
from graph_tsp_gcs import ProgramOptionsForGCSTSP

from util import timeit, INFO, WARN, ERROR, YAY  # INFO


from motion_planning_program import MotionPlanningProgram


class GraphTSPGCSProgram:
    """
    Choosing the order in which to move the blocks is a TSP problem.
    Choosing a collision free motion plan once an object is grasped is a GCS shortest path problem.
    This class combines the two: first we add all the TSP components, then all the MP components.
    """

    def __init__(
        self,
        tessellation_graph,
        graph_vertices: T.Dict[str, VertexTSP],
        graph_edges: T.Dict[str, EdgeTSP],
        num_possible_objects: int,
        program_options: ProgramOptionsForGCSTSP,
    ) -> None:

        self.num_possible_objects = num_possible_objects

        self.graph_vertices = graph_vertices
        self.graph_edges = graph_edges

        self.program_options = program_options

        # program vertices
        self.tessellation_graph = tessellation_graph
        self.vertices = dict()  # type: T.Dict[str, VertexTSPprogram]
        self.edges = dict()  # type: T.Dict[str, EdgeTSPprogram]
        self.prog = MathematicalProgram()
        self.solution = None

        # populate the program and vertex/edge dictionaries
        self.add_tsp_vertices_and_edges()
        self.add_tsp_variables_to_prog()
        self.add_tsp_constraints_to_prog()
        self.add_tsp_costs_to_prog()
        self.add_motion_planning()

        INFO(str(len(self.vertices)), " vertices", str(len(self.edges)), " edges")

        self.num_tsp_vertices = 0

    def add_program_vertex(self, tsp_vertex: VertexTSP):
        assert tsp_vertex.name not in self.vertices, (
            "Vertex with name " + tsp_vertex.name + " already exists"
        )
        self.vertices[tsp_vertex.name] = VertexTSPprogram.from_vertex_tsp(tsp_vertex)
        self.num_tsp_vertices += 1

    def add_program_tsp_edge(self, e: EdgeTSP):
        edge_name = e.name
        assert edge_name not in self.edges, "Edge " + edge_name + " already exists"
        self.edges[edge_name] = EdgeTSPprogram(
            self.vertices[e.left], self.vertices[e.right], edge_name
        )
        self.vertices[e.left].add_edge_out(edge_name)
        self.vertices[e.right].add_edge_in(edge_name)

    def add_tsp_vertices_and_edges(self) -> None:
        """
        Graph structure: add TSP vertices and edges.
        TSP vertices are start and target locations of blocks, start/target of arm.
        TSP edges are:
            from arm-start to any block-start
            from any block-target to any block-start (except its own)
            from any block-target to arm-target
        block-start to target-start is handled through motion planning.
        """
        ################################
        # add all vertices
        ################################
        for graph_vertex in self.graph_vertices:
            self.add_program_vertex(graph_vertex)

        for tsp_edge in self.graph_edges:
            self.add_program_tsp_edge(tsp_edge)

    def add_tsp_variables_to_prog(self) -> None:
        """
        Program variables -- add variables on vertices and edges.
        Though vertices have variables, really they correspond to left-right edge variables
        of a "motion planning edge".
        """
        ############################
        # vertex variables:
        #   visit variable v: n x 1 vector, 0/1 whether block-i is at start or at target
        #   visitation order variable order: number of tsp vertices visited so far
        for v in self.vertices.values():
            # visitation variable
            v.set_v(
                self.prog.NewContinuousVariables(
                    self.num_possible_objects, "v_" + v.name
                )
            )
            v.set_order(self.prog.NewContinuousVariables(1, "order_" + v.name)[0])

        ############################
        # edge variables:
        #   left and right visits
        #   left and right order
        #   flow
        for e in self.edges.values():
            # left and right visitation
            e.set_left_v(
                self.prog.NewContinuousVariables(
                    self.num_possible_objects, "left_v_" + e.name
                )
            )
            e.set_right_v(
                self.prog.NewContinuousVariables(
                    self.num_possible_objects, "right_v_" + e.name
                )
            )
            # left and right order
            e.set_left_order(
                self.prog.NewContinuousVariables(1, "left_order_" + e.name)[0]
            )
            e.set_right_order(
                self.prog.NewContinuousVariables(1, "right_order" + e.name)[0]
            )
            # add flow variable
            e.set_phi(self.prog.NewBinaryVariables(1, "phi_" + e.name)[0])

    def add_tsp_constraints_to_prog(self) -> None:
        """
        TSP constraints.
        These include:
            perspective machinery for left-right order edge variable
            perspective machinery for left-right visit edge variables
            regular set inclusion for order vertex variable
            regular set inclusion for visit vertex variable
            order in = order out, visit in = visit out
            order increase by 1 over active edge
            visits stay equal, visit[i] increases by 1 after i-th motion planning edge
            initial / terminal conditions
        """
        ############################
        # order and visit convex sets
        order_box = Box(lb=np.array([0]), ub=np.array([self.n - 1]), state_dim=1)
        visitation_box = Box(
            lb=np.zeros(self.num_possible_objects),
            ub=np.ones(self.num_possible_objects),
            state_dim=self.num_possible_objects,
        )
        ############################
        # add edge constraints:
        #   perspective set inclusion on (flow, order)
        #   perspective set inclusion on (flow, visit)
        for e in self.edges.values():
            # perspective constraints on order
            A, b = order_box.get_perspective_hpolyhedron_matrices()
            self.prog.AddLinearConstraint(le(A @ np.array([e.left_order, e.phi]), b))
            self.prog.AddLinearConstraint(le(A @ np.array([e.right_order, e.phi]), b))
            # perspective constraints on visits
            A, b = visitation_box.get_perspective_hpolyhedron_matrices()
            self.prog.AddLinearConstraint(le(A @ np.append(e.left_v, e.phi), b))
            self.prog.AddLinearConstraint(le(A @ np.append(e.right_v, e.phi), b))
            # increasing order
            self.prog.AddLinearConstraint(e.left_order + e.phi == e.right_order)
            # over all tsp edges, visit is same
            self.prog.AddLinearConstraint(eq(e.left_v, e.right_v))

        ############################
        # add vertex constraints:
        #   set inclusion on (order)
        #   order_in = order_out
        #   set inclusion on (visit)
        #   visit_in = visit_out
        #   order icnrease by 1 over any edge
        #   visitation equality over any edge
        #   visitation increase by 1 over "motion planning edge"
        for v in self.vertices.values():
            flow_in = sum([self.edges[e].phi for e in v.edges_in])
            flow_out = sum([self.edges[e].phi for e in v.edges_out])
            order_in = sum([self.edges[e].right_order for e in v.edges_in])
            order_out = sum([self.edges[e].left_order for e in v.edges_out])
            v_in = sum([self.edges[e].right_v for e in v.edges_in])
            v_out = sum([self.edges[e].left_v for e in v.edges_out])
            if v.name == "start":
                # it's the start vertex; initial conditions
                # flow out is 1
                self.prog.AddLinearConstraint(flow_out == 1)
                # order at vertex is 0
                self.prog.AddLinearConstraint(v.order == 0)
                # order continuity: order_out is 0
                self.prog.AddLinearConstraint(v.order == order_out)
                # 0 visits have been made yet
                self.prog.AddLinearConstraint(
                    eq(v.v, np.zeros(self.num_possible_objects))
                )
                # visit continuity
                self.prog.AddLinearConstraint(eq(v.v, v_out))
            elif v.name == "target":
                # it's the target vertex; final conditions
                # flow in is 1
                self.prog.AddLinearConstraint(flow_in == 1)
                # order at vertex is n-1
                self.prog.AddLinearConstraint(v.order == self.n - 1)
                # order continuity: order in is n-1
                self.prog.AddLinearConstraint(v.order == order_in)
                # all blocks have been visited
                self.prog.AddLinearConstraint(
                    eq(v.v, np.ones(self.num_possible_objects))
                )
                # visit continuity: v me is v in
                self.prog.AddLinearConstraint(eq(v.v, v_in))
            elif v.name[0] == "s":
                # it's a start block vertex
                # flow in is 1
                self.prog.AddLinearConstraint(
                    flow_in == 1
                )  # flow out is set in motion planning
                # vertex order is sum of orders in
                self.prog.AddLinearConstraint(v.order == order_in)
                # vertex visit is sum of visits in
                self.prog.AddLinearConstraint(eq(v.v, v_in))
                # order belongs to a set (TODO: redundant?)
                A, b = order_box.get_hpolyhedron_matrices()
                self.prog.AddLinearConstraint(le(A @ np.array([v.order]), b))
                # visitations belong to a set (TODO: redundant?)
                A, b = visitation_box.get_hpolyhedron_matrices()
                self.prog.AddLinearConstraint(le(A @ v.v, b))
            elif v.name[0] == "t":
                # it's a target block vertex
                # flow out is 1
                self.prog.AddLinearConstraint(
                    flow_out == 1
                )  # flow in is set in motion planning
                # vertex order is sum of orders out
                self.prog.AddLinearConstraint(v.order == order_out)
                # vertex visit is sum of visits out
                self.prog.AddLinearConstraint(eq(v.v, v_out))
                # order belongs to a set (TODO: redundant?)
                A, b = order_box.get_hpolyhedron_matrices()
                self.prog.AddLinearConstraint(le(A @ np.array([v.order]), b))
                # visitations belong to a set (TODO: redundant?)
                A, b = visitation_box.get_hpolyhedron_matrices()
                self.prog.AddLinearConstraint(le(A @ v.v, b))

                # order / visitation continuity over motion planning
                # order at t_i_tsp = order at v_i_tsp + 1
                sv = self.vertices["s" + v.name[1:]]
                assert sv.block_index == v.block_index, "block indeces do not match"
                self.prog.AddLinearConstraint(sv.order + 1 == v.order)
                # visitations hold except for the block i at which we are in
                for i in range(self.num_possible_objects):
                    if i == v.block_index:
                        self.prog.AddLinearConstraint(sv.v[i] + 1 == v.v[i])
                    else:
                        self.prog.AddLinearConstraint(sv.v[i] == v.v[i])

    def add_tsp_costs_to_prog(self) -> None:
        """
        TSP costs are constants: pay a fixed price for going from target of last to start of next.
        """
        for e in self.edges.values():
            e.set_cost(np.linalg.norm(e.right.block_position - e.left.block_position))
        self.prog.AddLinearCost(sum([e.phi * e.cost for e in self.edges.values()]))

    # TODO
    # this should be called "add gcs edge"
    def add_motion_planning(self) -> None:
        """
        Adding motion planning edges, vertices, constraints, costs for each "motion planning edge".
        """
        for gcs_edge in self.gcs_edges:
            MotionPlanningProgram(
                prog=self.prog,
                vertices=self.vertices,
                edges=self.edges,
                start_vertex=None,
                target_vertex=None,
                tessellation_graph=self.tessellation_graph,  # already a graph?
                options=self.program_options,
            )

    def solve(self):
        """Solve the program"""
        x = timeit()
        self.solution = Solve(self.prog)
        x.dt("Solving the program")
        if self.solution.is_success():
            YAY("Optimal primal cost is %.5f" % self.solution.get_optimal_cost())
        else:
            ERROR("PRIMAL SOLVE FAILED!")
            ERROR("Optimal primal cost is %.5f" % self.solution.get_optimal_cost())
            raise Exception("Still ways to go till we solve all of robotics, mate")

        flows = [self.solution.GetSolution(e.phi) for e in self.edges.values()]
        not_tight = np.any(
            np.logical_and(0.01 < np.array(flows), np.array(flows) < 0.99)
        )
        if self.convex_relaxation:
            if not_tight:
                WARN("CONVEX RELAXATION NOT TIGHT")
            else:
                YAY("CONVEX RELAXATION IS TIGHT")
        else:
            YAY("WAS SOLVING INTEGER PROGRAM")

        flow_vars = [(e, self.solution.GetSolution(e.phi)) for e in self.edges.values()]
        non_zero_edges = [e for (e, flow) in flow_vars if flow > 0.01]
        # print(self.solution.GetSolution(self.vertices["s1_tsp"].v))
        # for (e, flow) in flow_vars:
        #     if 0.01 < flow < 0.99:
        #         print(e.name, flow)

        v_path, e_path = self.find_path_to_target(
            non_zero_edges, self.vertices["start"]
        )
        now_pose = self.start_pos.copy()

    def get_trajectory_for_drawing(self) -> T.Tuple[npt.NDArray, T.List[str]]:
        """Returns modes and positions for Draw2DSolution class"""

        def add_me(pose: npt.NDArray, mode: str):
            p = pose.copy()
            p.resize(p.size)
            poses.append(p)
            modes.append(mode)

        flow_vars = [(e, self.solution.GetSolution(e.phi)) for e in self.edges.values()]
        non_zero_edges = [e for (e, flow) in flow_vars if flow > 0.01]
        v_path, e_path = self.find_path_to_target(
            non_zero_edges, self.vertices["start"]
        )
        now_pose = self.start_pos.copy()
        poses, modes = [], []
        i, mode = 0, "0"
        # initial state
        add_me(now_pose, mode)
        # go through all vertices in the path
        while i < len(v_path):
            # it's a TSP node, arm is free
            if v_path[i].value is not None:
                now_pose[0], mode = v_path[i].value, "0"
                # if next node is an MP node -- grasp
                if i + 1 < len(v_path) and v_path[i + 1].value is None:
                    mode = "1"
            # it's an MP node, grasp
            else:
                now_pose[0], mode = self.solution.GetSolution(e_path[i].right_pos), "1"
                now_pose[v_path[i].block_index + 1] = now_pose[0]
            add_me(now_pose, mode)
            i += 1
        return np.array(poses), modes

    def find_path_to_target(
        self, edges: T.List[EdgeTSPprogram], start: VertexTSPprogram
    ) -> T.Tuple[T.List[VertexTSPprogram], T.List[EdgeTSPprogram]]:
        """Given a set of active edges, find a path from start to target recursively"""
        edges_out = [e for e in edges if e.left == start]
        assert (
            len(edges_out) == 1
        ), "More than one edge flowing out of the vertex, it's not a path!"
        current_edge = edges_out[0]
        v = current_edge.right
        # if target reached - return, else -- recursion
        target_reached = v.name == "target"
        if target_reached:
            return [start] + [v], [current_edge]
        else:
            v, e = self.find_path_to_target(edges, v)
            return [start] + v, [current_edge] + e
