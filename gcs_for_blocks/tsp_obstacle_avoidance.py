import typing as T

import numpy as np
import numpy.typing as npt

from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module
    MathematicalProgram,
    Solve,
)
from pydrake.math import le, eq  # pylint: disable=import-error, no-name-in-module

from .util import timeit, INFO, WARN, ERROR, YAY  # INFO
from .axis_aligned_set_tesselation import (
    Box,
    AlignedSet,
    axis_aligned_tesselation,
    locations_to_aligned_sets,
    get_obstacle_to_set_mapping,
)
from .tsp_vertex_edge import Vertex, Edge
from .motion_planning_obstacles_on_off import MotionPlanning


class BlockMovingObstacleAvoidance:
    """
    Choosing the order in which to move the blocks is a TSP problem.
    Choosing a collision free motion plan once an object is grasped is a GCS shortest path problem.
    This class combines the two: first we add all the TSP components, then all the MP components.
    """

    def __init__(
        self,
        start_pos: T.List[T.Tuple[float]],
        target_pos: T.List[T.Tuple[float]],
        bounding_box: AlignedSet,
        block_width: float = 1.0,
        convex_relaxation: bool = False,
        share_edge_tol: float = 0.000001,
    ) -> None:
        self.num_blocks = len(start_pos) - 1  # type: int
        assert len(target_pos) == len(start_pos)
        # all position vectors
        self.start_pos = np.array(start_pos)
        self.target_pos = np.array(target_pos)
        self.start_arm_pos = np.array(start_pos[0])
        self.target_arm_pos = np.array(target_pos[0])
        self.start_block_pos = [np.array(x) for x in start_pos[1:]]
        self.target_block_pos = [np.array(x) for x in target_pos[1:]]
        # start and target vertecis
        self.start = "sa_tsp"  # str
        self.target = "ta_tsp"  # str
        self.bounding_box = bounding_box  # type: AlignedSet
        # make a tesselation
        obstacles = locations_to_aligned_sets(
            self.start_block_pos, self.target_block_pos, block_width, self.bounding_box
        )
        self.convex_set_tesselation = axis_aligned_tesselation(
            bounding_box.copy(), obstacles
        )
        self.obstacle_to_set = get_obstacle_to_set_mapping(
            self.start_block_pos, self.target_block_pos, self.convex_set_tesselation
        )
        self.share_edge_tol = share_edge_tol
        # init the program
        self.vertices = dict()  # type: T.Dict[str, Vertex]
        self.edges = dict()  # type: T.Dict[str, Edge]
        self.convex_relaxation = convex_relaxation
        self.prog = MathematicalProgram()
        self.solution = None
        # populate the program and vertex/edge dictionaries
        self.add_tsp_vertices_and_edges()
        self.add_tsp_variables_to_prog()
        self.add_tsp_constraints_to_prog()
        self.add_tsp_costs_to_prog()
        self.add_motion_planning()

        INFO(str(len(self.vertices)), " vertices", str(len(self.edges)), " edges")

    @property
    def n(self) -> int:
        """Number of vertices"""
        return 2 * (self.num_blocks + 1)

    def s(self, name: str) -> str:
        """Name a start-block vertex"""
        return "s" + str(name) + "_tsp"

    def t(self, name: str) -> str:
        """Name a target-block vertex"""
        return "t" + str(name) + "_tsp"

    def add_vertex(self, name: str, value: npt.NDArray, block_index: int) -> None:
        """Add vertex to the respective dictionary"""
        assert name not in self.vertices, "Vertex with name " + name + " already exists"
        self.vertices[name] = Vertex(name, value, block_index)

    def add_edge(self, left_name: str, right_name: str) -> None:
        """Add edge to the respective dictionary"""
        edge_name = left_name + "_" + right_name
        assert edge_name not in self.edges, "Edge " + edge_name + " already exists"
        self.edges[edge_name] = Edge(
            self.vertices[left_name], self.vertices[right_name], edge_name
        )
        self.vertices[left_name].add_edge_out(edge_name)
        self.vertices[right_name].add_edge_in(edge_name)

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
        # add start/target arm vertices
        self.add_vertex(self.start, self.start_arm_pos, -1)
        self.add_vertex(self.target, self.target_arm_pos, -1)
        # add start/target block vertices
        for i, pos in enumerate(self.start_block_pos):
            self.add_vertex(self.s(i), pos, i)
        for i, pos in enumerate(self.target_block_pos):
            self.add_vertex(self.t(i), pos, i)

        ################################
        # add all edges
        ################################
        # add edge to from initial arm location to final arm location
        self.add_edge(self.s("a"), self.t("a"))
        for j in range(self.num_blocks):
            # from start to any
            self.add_edge(self.s("a"), self.s(j))
            # from any to target
            self.add_edge(self.t(j), self.t("a"))
            # from any to target to any start
            for i in range(self.num_blocks):
                if i != j:
                    self.add_edge(self.t(j), self.s(i))
            # from start block to target block is motion planning!

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
            v.set_v(self.prog.NewContinuousVariables(self.num_blocks, "v_" + v.name))
            v.set_order(self.prog.NewContinuousVariables(1, "order_" + v.name)[0])

        ############################
        # edge variables:
        #   left and right visits
        #   left and right order
        #   flow
        for e in self.edges.values():
            # left and right visitation
            e.set_left_v(
                self.prog.NewContinuousVariables(self.num_blocks, "left_v_" + e.name)
            )
            e.set_right_v(
                self.prog.NewContinuousVariables(self.num_blocks, "right_v_" + e.name)
            )
            # left and right order
            e.set_left_order(
                self.prog.NewContinuousVariables(1, "left_order_" + e.name)[0]
            )
            e.set_right_order(
                self.prog.NewContinuousVariables(1, "right_order" + e.name)[0]
            )

            # add flow variable
            # if self.convex_relaxation:
            #     e.set_phi(self.prog.NewContinuousVariables(1, "phi_" + e.name)[0])
            #     self.prog.AddLinearConstraint(e.phi, 0.0, 1.0)
            # else:
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
            lb=np.zeros(self.num_blocks),
            ub=np.ones(self.num_blocks),
            state_dim=self.num_blocks,
        )
        ############################
        # add edge constraints:
        #   perspective set inclusion on (flow, order)
        #   perspective set inclusion on (flow, visit)
        for e in self.edges.values():
            # perspective constraints on order
            A, b = order_box.get_perspective_hpolyhedron()
            self.prog.AddLinearConstraint(le(A @ np.array([e.left_order, e.phi]), b))
            self.prog.AddLinearConstraint(le(A @ np.array([e.right_order, e.phi]), b))
            # perspective constraints on visits
            A, b = visitation_box.get_perspective_hpolyhedron()
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
            if v.name == self.start:
                # it's the start vertex; initial conditions
                # flow out is 1
                self.prog.AddLinearConstraint(flow_out == 1)
                # order at vertex is 0
                self.prog.AddLinearConstraint(v.order == 0)
                # order continuity: order_out is 0
                self.prog.AddLinearConstraint(v.order == order_out)
                # 0 visits have been made yet
                self.prog.AddLinearConstraint(eq(v.v, np.zeros(self.num_blocks)))
                # visit continuity
                self.prog.AddLinearConstraint(eq(v.v, v_out))
            elif v.name == self.target:
                # it's the target vertex; final conditions
                # flow in is 1
                self.prog.AddLinearConstraint(flow_in == 1)
                # order at vertex is n-1
                self.prog.AddLinearConstraint(v.order == self.n - 1)
                # order continuity: order in is n-1
                self.prog.AddLinearConstraint(v.order == order_in)
                # all blocks have been visited
                self.prog.AddLinearConstraint(eq(v.v, np.ones(self.num_blocks)))
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
                A, b = order_box.get_hpolyhedron()
                self.prog.AddLinearConstraint(le(A @ np.array([v.order]), b))
                # visitations belong to a set (TODO: redundant?)
                A, b = visitation_box.get_hpolyhedron()
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
                A, b = order_box.get_hpolyhedron()
                self.prog.AddLinearConstraint(le(A @ np.array([v.order]), b))
                # visitations belong to a set (TODO: redundant?)
                A, b = visitation_box.get_hpolyhedron()
                self.prog.AddLinearConstraint(le(A @ v.v, b))

                # order / visitation continuity over motion planning
                # order at t_i_tsp = order at v_i_tsp + 1
                sv = self.vertices["s" + v.name[1:]]
                assert sv.block_index == v.block_index, "block indeces do not match"
                self.prog.AddLinearConstraint(sv.order + 1 == v.order)
                # visitations hold except for the block i at which we are in
                for i in range(self.num_blocks):
                    if i == v.block_index:
                        self.prog.AddLinearConstraint(sv.v[i] + 1 == v.v[i])
                    else:
                        self.prog.AddLinearConstraint(sv.v[i] == v.v[i])

    def add_tsp_costs_to_prog(self) -> None:
        """
        TSP costs are constants: pay a fixed price for going from target of last to start of next.
        """
        for e in self.edges.values():
            e.cost = np.linalg.norm(e.right.value - e.left.value)
        self.prog.AddLinearCost(sum([e.phi * e.cost for e in self.edges.values()]))

    def add_motion_planning(self) -> None:
        """
        Adding motion planning edges, vertices, constraints, costs for each "motion planning edge".
        """
        for block_index in range(self.num_blocks):
            MotionPlanning(
                prog=self.prog,
                all_vertices=self.vertices,
                all_edges=self.edges,
                start_block_pos=self.start_block_pos,
                target_block_pos=self.target_block_pos,
                convex_set_tesselation=self.convex_set_tesselation,
                obstacle_to_set=self.obstacle_to_set,
                moving_block_index=block_index,
                convex_relaxation=self.convex_relaxation,
                share_edge_tol=self.share_edge_tol,
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
            non_zero_edges, self.vertices[self.start]
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
            non_zero_edges, self.vertices[self.start]
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
        self, edges: T.List[Edge], start: Vertex
    ) -> T.Tuple[T.List[Vertex], T.List[Edge]]:
        """Given a set of active edges, find a path from start to target recursively"""
        edges_out = [e for e in edges if e.left == start]
        assert (
            len(edges_out) == 1
        ), "More than one edge flowing out of the vertex, it's not a path!"
        current_edge = edges_out[0]
        v = current_edge.right
        # if target reached - return, else -- recursion
        target_reached = v.name == self.target
        if target_reached:
            return [start] + [v], [current_edge]
        else:
            v, e = self.find_path_to_target(edges, v)
            return [start] + v, [current_edge] + e
