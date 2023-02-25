import typing as T

import numpy as np
import numpy.typing as npt

from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module
    MathematicalProgram,
    Solve,
)
from pydrake.math import le, eq  # pylint: disable=import-error, no-name-in-module

from .axis_aligned_set_tesselation import Box
from .tsp_vertex_edge import Vertex, Edge
from .util import timeit, INFO, WARN, ERROR, YAY


class TSPasGCS:
    def __init__(self):
        self.edges = dict()  # type: T.Dict[str, Edge]
        self.vertices = dict()  # type: T.Dict[str, Vertex]
        self.start = None  # type: str # name of the start node
        self.target = None  # type: str # name of the target node
        self.primal_prog = None  # type: MathematicalProgram
        self.primal_solution = None
        self.convex_relaxation = None  # type: bool

    @property
    def n(self):  # number of vertices
        return len(self.vertices)

    def add_vertex(self, name: str, value: npt.NDArray):
        # value is ~ needed for calculating the edge pathd
        assert name not in self.vertices, "Vertex with name " + name + " already exists"
        self.vertices[name] = Vertex(name, value)

    def add_edge(self, left_name: str, right_name: str, cost: float = None):
        edge_name = left_name + "_" + right_name
        assert edge_name not in self.edges, "Edge " + edge_name + " already exists"
        self.edges[edge_name] = Edge(
            self.vertices[left_name], self.vertices[right_name], edge_name, cost
        )
        self.vertices[left_name].add_edge_out(edge_name)
        self.vertices[right_name].add_edge_in(edge_name)

    def add_cost_on_edge(self, edge_name: str, cost: float):
        self.edges[edge_name].set_cost(cost)

    def build_dual_optimization_program(self):
        raise Exception("Not Implemented")

    def set_start_target(self, start_name: str, target_name: str):
        self.start = start_name
        self.target = target_name

    def build_primal_optimization_program(self, convex_relaxation=True):
        assert self.start is not None
        assert self.target is not None
        assert self.start in self.vertices
        assert self.target in self.vertices

        self.primal_prog = MathematicalProgram()
        self.convex_relaxation = convex_relaxation

        self.add_decision_variables_to_primal()
        self.add_constraints_to_primal()
        self.add_costs_to_primal()

    def add_decision_variables_to_primal(self):
        # for each edge
        for e in self.edges.values():
            # add left / right order variables
            e.set_left_order(
                self.primal_prog.NewContinuousVariables(1, "left_order_" + e.name)[0]
            )
            e.set_right_order(
                self.primal_prog.NewContinuousVariables(1, "right_order_" + e.name)[0]
            )
            # add flow variable
            if self.convex_relaxation:
                if e.right.name == self.target:
                    e.set_phi(
                        self.primal_prog.NewBinaryVariables(1, "phi_" + e.name)[0]
                    )
                else:
                    e.set_phi(
                        self.primal_prog.NewContinuousVariables(1, "phi_" + e.name)[0]
                    )
                    self.primal_prog.AddLinearConstraint(e.phi, 0.0, 1.0)
            else:
                e.set_phi(self.primal_prog.NewBinaryVariables(1, "phi_" + e.name)[0])

    def add_constraints_to_primal(self):
        # perspective polyhedron constraints on start nodes
        s_order_box = Box(lb=np.array([1]), ub=np.array([self.n - 1]), state_dim=1)
        A1, b1 = s_order_box.get_perspective_hpolyhedron()
        # perspective polyhedron constraints on target nodes
        t_order_box = Box(lb=np.array([2]), ub=np.array([self.n - 1]), state_dim=1)
        A2, b2 = t_order_box.get_perspective_hpolyhedron()
        ####################################
        # for each edge
        for e in self.edges.values():
            ####################################
            # add constraints on left order
            ####################################
            if e.left.name == self.start:
                # left order of any edge from origin is 0
                self.primal_prog.AddLinearConstraint(e.left_order == 0)
            elif e.left.name[0] == "s":
                # all start nodes have order of at least 1
                self.primal_prog.AddLinearConstraint(
                    le(A1 @ np.array([e.left_order, e.phi]), b1)
                )
            elif e.left.name[0] == "t":
                # all target nodes have order of at least 2
                self.primal_prog.AddLinearConstraint(
                    le(A2 @ np.array([e.left_order, e.phi]), b2)
                )

            ####################################
            # add constraints on right order
            ####################################
            # redundant?
            if e.right.name[0] == "s":
                # all start nodes have order of at least 1
                self.primal_prog.AddLinearConstraint(
                    le(A1 @ np.array([e.right_order, e.phi]), b1)
                )
            else:
                # all target nodes have order of at least 2
                self.primal_prog.AddLinearConstraint(
                    le(A2 @ np.array([e.right_order, e.phi]), b2)
                )

            # order increase constraint
            self.primal_prog.AddLinearConstraint(e.left_order + e.phi == e.right_order)

        ####################################
        # for each vertex
        for v in self.vertices.values():
            # if not start -- add "flow in is 1" constraint
            if v.name != self.start:
                flow_in = sum([self.edges[e].phi for e in v.edges_in])
                self.primal_prog.AddLinearConstraint(flow_in == 1)
            # if not target -- add "flow out is 1" constraint
            if v.name != self.target:
                flow_out = sum([self.edges[e].phi for e in v.edges_out])
                self.primal_prog.AddLinearConstraint(flow_out == 1)

            # order in = order out
            order_out = sum([self.edges[e].left_order for e in v.edges_out])
            order_in = sum([self.edges[e].right_order for e in v.edges_in])
            # start -- order out is 0
            if v.name == self.start:
                self.primal_prog.AddLinearConstraint(order_out == 0.0)
            # target -- order in is n-1
            elif v.name == self.target:
                self.primal_prog.AddLinearConstraint(order_in == self.n - 1)
            # order in = order out
            else:
                self.primal_prog.AddLinearConstraint(order_out == order_in)

        # left and right vertices
        left_vertices = ["s0"] + ["t" + str(i) for i in range(1, int(self.n / 2))]
        right_vertices = ["t0"] + ["s" + str(i) for i in range(1, int(self.n / 2))]

        # total order is sum from 0 to n-1
        total_order_sum = (self.n - 1) * (self.n) / 2
        # left order is sum of even values from 0 to n-1 = 2 * sum from 0 to (n-1)/2
        left_order_sum = (self.n / 2 - 1) * (self.n / 2)
        # right order is full order - left order
        right_order_sum = total_order_sum - left_order_sum

        self.primal_prog.AddLinearConstraint(
            sum(
                [
                    e.right_order
                    for e in self.edges.values()
                    if e.right.name in left_vertices
                ]
            )
            == left_order_sum
        )
        self.primal_prog.AddLinearConstraint(
            sum(
                [
                    e.left_order
                    for e in self.edges.values()
                    if e.left.name in left_vertices
                ]
            )
            == left_order_sum
        )
        self.primal_prog.AddLinearConstraint(
            sum(
                [
                    e.right_order
                    for e in self.edges.values()
                    if e.right.name in right_vertices
                ]
            )
            == right_order_sum
        )
        self.primal_prog.AddLinearConstraint(
            sum(
                [
                    e.left_order
                    for e in self.edges.values()
                    if e.left.name in right_vertices
                ]
            )
            == right_order_sum - (self.n - 1)
        )

        # self.primal_prog.AddLinearConstraint(self.edges["s0_s6"].phi==1)

        # total sum is given; don't need it if i already sum up left/right individually
        # self.primal_prog.AddLinearConstraint( sum( [e.right_order for e in self.edges.values()]) == (self.n-1)*self.n/2 )
        # self.primal_prog.AddLinearConstraint( sum( [e.left_order for e in self.edges.values()]) == (self.n-2)*(self.n-1)/2 )
        # k = int(self.n/2)
        # for j in range(1,k-1):
        #     set1 = ["s0"] + ["t" + str(i) for i in range(1,j+1)]
        #     set2 = ["s"+str(i) for i in range(j+1,k)]
        #     full_sum = sum([e.phi for e in self.edges.values() if (e.left.name in set1 and e.right.name in set2) ])
        #     self.primal_prog.AddLinearConstraint(full_sum >= 1)

    def add_costs_to_primal(self):
        self.primal_prog.AddLinearCost(
            sum([e.phi * e.cost for e in self.edges.values()])
        )

    def solve_primal(self, verbose=False):
        # solve
        x = timeit()
        self.primal_solution = Solve(self.primal_prog)
        x.dt("Solving the program")

        if self.primal_solution.is_success():
            YAY("Optimal primal cost is %.5f" % self.primal_solution.get_optimal_cost())
        else:
            ERROR("PRIMAL SOLVE FAILED!")
            ERROR(
                "Optimal primal cost is %.5f" % self.primal_solution.get_optimal_cost()
            )
            return

        if self.convex_relaxation:
            flows = [
                self.primal_solution.GetSolution(e.phi) for e in self.edges.values()
            ]
            not_tight = np.any(
                np.logical_and(0.01 < np.array(flows), np.array(flows) < 0.99)
            )
            if not_tight:
                WARN("CONVEX RELAXATION NOT TIGHT")
            else:
                YAY("CONVEX RELAXATION IS TIGHT")
        else:
            YAY("WAS SOLVING INTEGER PROGRAM")

        if verbose:
            self.verbose_solution()

    def verbose_solution(self):
        np.set_printoptions(precision=3)
        flow_vars = [
            (e.name, self.primal_solution.GetSolution(e.phi), e.cost)
            for e in self.edges.values()
        ]
        for (name, flow, cost) in flow_vars:
            if flow > 0.01 and name[0] != "s":
                print(name, f"{flow:03f}", cost, f"{flow*cost:03f}")
            if flow > 0.01 and name[0:3] == "s0_":
                print(name, f"{flow:03f}", cost, f"{flow*cost:03f}")

        pots = []
        for v in self.vertices.values():
            # sum_of_y = [self.primal_solution.GetSolution(self.edges[e].left_order) for e in v.edges_out]
            # sum_of_z = [self.primal_solution.GetSolution(self.edges[e].right_order) for e in v.edges_in]
            # print(v.name, "left order", sum_of_y, "right order", sum_of_z)
            sum_of_y = sum(
                [
                    self.primal_solution.GetSolution(self.edges[e].left_order)
                    for e in v.edges_out
                ]
            )
            sum_of_z = sum(
                [
                    self.primal_solution.GetSolution(self.edges[e].right_order)
                    for e in v.edges_in
                ]
            )
            pots.append((v.name, "order", sum_of_z))
        # pots = [name for (name, _, _) in sorted(pots, key = lambda x: x[1])]
        pots = [x for x in sorted(pots, key=lambda x: x[1])]
        for x in pots:
            print(x)
        # print(pots)


def build_block_moving_gcs_tsp(
    start: T.List[T.Tuple[float]], target: T.List[T.Tuple[float]]
) -> TSPasGCS:
    num_objects = len(target)
    assert len(start) == len(target)
    # naming
    s = lambda i: "s" + str(i)  # start node name
    t = lambda i: "t" + str(i)  # target node name
    e = lambda i, j: i + "_" + j  # edge name

    gcs = TSPasGCS()

    # add all vertices
    for i in range(num_objects):
        gcs.add_vertex(s(i), np.array(start[i]))
        gcs.add_vertex(t(i), np.array(target[i]))

    # add edge to from initial arm location to final arm location
    gcs.add_edge(s(0), t(0), np.linalg.norm(start[0] - target[0]))
    for i in range(1, num_objects):
        # start is connected to any object start locations
        gcs.add_edge(s(0), s(i), np.linalg.norm(start[0] - start[i]))
        # after ungrasping any object, we can move to arm target location
        gcs.add_edge(t(i), t(0), np.linalg.norm(target[i] - target[0]))
        # once we pick up an object, we must move it to the goal
        gcs.add_edge(s(i), t(i), np.linalg.norm(start[i] - target[i]))
        # after ungrasping an object, we can go and pick up any other object
        for j in range(1, num_objects):
            if i != j:
                gcs.add_edge(t(i), s(j), np.linalg.norm(target[i] - start[j]))

    # set start and target
    gcs.set_start_target("s0", "t0")
    return gcs
