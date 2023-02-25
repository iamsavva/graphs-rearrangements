# pyright: reportMissingImports=false
import typing as T

import numpy as np
import numpy.typing as npt

import pydot
from tqdm import tqdm
from IPython.display import Image, display
import time

# from PIL import Image as PIL_Image

import pydrake.geometry.optimization as opt  # pylint: disable=import-error
from pydrake.geometry.optimization import (  # pylint: disable=import-error
    Point,
    GraphOfConvexSets,
    HPolyhedron,
    ConvexSet,
)
from pydrake.solvers import (  # pylint: disable=import-error, unused-import
    Binding,
    L2NormCost,
    LinearConstraint,
    LinearEqualityConstraint,
    LinearCost,
)

from .util import ERROR, WARN, INFO, YAY
from .gcs_options import GCSforAutonomousBlocksOptions, EdgeOptAB
from .set_tesselation_2d import SetTesselation
from .gcs import GCSforBlocks


class GCSAutonomousBlocks(GCSforBlocks):
    """
    GCS for N-dimensional block moving using a top-down suction cup.
    """

    ###################################################################################
    # Building the finite horizon GCS

    def __init__(self, options: GCSforAutonomousBlocksOptions):
        # options
        self.opt = options

        # init the graph
        self.gcs = GraphOfConvexSets()
        self.graph_built = False
        self.solution = None

        self.set_gen = SetTesselation(options)

        # name to vertex dictionary, populated as we populate the graph with vertices
        self.name_to_vertex = dict()  # T.Dict[str, GraphOfConvexSets.Vertex]
        print("finished init")

    def build_the_graph_simple(
        self,
        start_state: Point,
        target_state: Point,
    ) -> None:
        """
        Build the GCS graph of horizon H from start to target nodes.
        TODO:
        - allow target state to be a set
        """
        # reset the graph
        self.gcs = GraphOfConvexSets()
        self.graph_built = False

        # add all edges
        self.add_everything(start_state, target_state)

        self.graph_built = True

    ###################################################################################
    # Adding layers of nodes (trellis diagram style)

    def add_everything(self, start_state: Point, target_state: Point) -> None:
        self.add_vertex(start_state, "start")
        self.add_vertex(target_state, "target")

        ############################
        start_set_string = self.set_gen.construct_rels_representation_from_point(
            start_state.x()
        )
        start_set = self.set_gen.rels2set[start_set_string]
        self.add_vertex(start_set, start_set_string)
        self.connect_vertices("start", start_set_string, EdgeOptAB.equality_edge())

        target_set_string = self.set_gen.construct_rels_representation_from_point(
            target_state.x()
        )
        target_set = self.set_gen.rels2set[target_set_string]
        self.add_vertex(target_set, target_set_string)
        self.connect_vertices(target_set_string, "target", EdgeOptAB.target_edge())

        num_edges = 2

        if self.opt.edge_gen == "all":
            for rels in self.set_gen.rels2set:
                self.add_vertex(self.set_gen.rels2set[rels], rels)
                nbhd = self.set_gen.get_1_step_neighbours(rels)

                for nbh in nbhd:
                    self.add_vertex(self.set_gen.rels2set[nbh], nbh)
                    self.connect_vertices(rels, nbh, EdgeOptAB.move_edge())
                    num_edges += 1

        elif self.opt.edge_gen == "binary_tree_down":
            already_added = set()
            already_added.add(start_set_string)
            frontier = set()
            frontier.add(start_set_string)
            next_frontier = set()

            while target_set_string not in already_added:
                for f in frontier:
                    # find neighbours of f

                    # nbhd = self.set_gen.get_1_step_neighbours(f)
                    nbhd = self.set_gen.get_useful_1_step_neighbours(
                        f, target_set_string
                    )
                    for nbh in nbhd:
                        # for each neighbour: if it's not in current / previous layers -- add it
                        if nbh not in already_added:
                            if nbh in self.set_gen.rels2set:
                                self.add_vertex(self.set_gen.rels2set[nbh], nbh)
                                self.connect_vertices(f, nbh, EdgeOptAB.move_edge())
                                next_frontier.add(nbh)
                                num_edges += 1

                frontier = next_frontier.copy()
                already_added = already_added.union(next_frontier)
                next_frontier = set()
        else:
            raise Exception("Inapproprate edge gen: " + self.opt.edge_gen)

        print("num edges is ", num_edges)

    ###################################################################################
    # Populating edges and vertices

    def add_edge(
        self,
        left_vertex: GraphOfConvexSets.Vertex,
        right_vertex: GraphOfConvexSets.Vertex,
        edge_opt: EdgeOptAB,
    ) -> None:
        """
        READY
        Add an edge between two vertices, as well as corresponding constraints and costs.
        """
        # add an edge
        edge_name = self.get_edge_name(left_vertex.name(), right_vertex.name())
        edge = self.gcs.AddEdge(left_vertex, right_vertex, edge_name)

        # -----------------------------------------------------------------
        # Adding constraints
        # -----------------------------------------------------------------
        if edge_opt.add_set_transition_constraint:
            left_set = self.set_gen.rels2set[left_vertex.name()]
            self.add_common_set_at_transition_constraint(left_set, edge)
        if edge_opt.add_equality_constraint:
            self.add_point_equality_constraint(edge)
        # -----------------------------------------------------------------
        # Adding costs
        # -----------------------------------------------------------------
        # add movement cost on the edge
        if edge_opt.add_each_block_movement_cost:
            self.add_each_block_movement_cost(edge)
            # self.add_full_movement_cost(edge)

    ###################################################################################
    # Adding constraints and cost terms

    def add_common_set_at_transition_constraint(
        self, left_vertex_set: HPolyhedron, edge: GraphOfConvexSets.Edge
    ) -> None:
        """
        READY
        Add a constraint that the right vertex belongs to the same mode as the left vertex
        """
        # fill in linear constraint on the right vertex
        A = left_vertex_set.A()
        lb = -np.ones(left_vertex_set.b().size) * 1000
        ub = left_vertex_set.b()
        set_con = LinearConstraint(A, lb, ub)
        edge.AddConstraint(Binding[LinearConstraint](set_con, edge.xv()))

    def add_each_block_movement_cost(self, edge: GraphOfConvexSets.Edge) -> None:
        xu, xv = edge.xu(), edge.xv()
        for i in range(self.opt.num_blocks):
            d = self.opt.block_dim
            n = self.opt.state_dim
            A = np.zeros((d, 2 * n))
            A[:, i * d : i * d + d] = np.eye(d)
            A[:, n + i * d : n + i * d + d] = -np.eye(d)
            b = np.zeros(d)
            # add the cost
            cost = L2NormCost(A, b)
            edge.AddCost(Binding[L2NormCost](cost, np.append(xv, xu)))

    def add_full_movement_cost(self, edge):
        xu, xv = edge.xu(), edge.xv()
        n = self.opt.state_dim
        A = np.hstack((np.eye(n), -np.eye(n)))
        b = np.zeros(n)
        # add the cost
        cost = L2NormCost(A, b)
        edge.AddCost(Binding[L2NormCost](cost, np.append(xv, xu)))

    ###################################################################################

    def get_edge_name(self, left_vertex_name: str, right_vertex_name: str) -> str:
        return left_vertex_name + "_" + right_vertex_name

    ###################################################################################
    # Solve and display solution

    # def get_solution_path(self) -> T.Tuple[T.List[str], npt.NDArray]:
    #     """Given a solved GCS problem, and assuming it's tight, find a path from start to target"""
    #     assert self.graph_built, "Must build graph first!"
    #     assert self.solution.is_success(), "Solution was not found"
    #     # find edges with non-zero flow
    #     flow_variables = [e.phi() for e in self.gcs.Edges()]
    #     flow_results = [self.solution.GetSolution(p) for p in flow_variables]

    #     not_tight = np.any(
    #         np.logical_and(0.05 < np.array(flow_results), np.array(flow_results) < 0.95)
    #     )
    #     if not_tight:
    #         WARN("CONVEX RELAXATION NOT TIGHT")

    #     active_edges = [
    #         edge for edge, flow in zip(self.gcs.Edges(), flow_results) if flow >= 0.99
    #     ]
    #     # using these edges, find the path from start to target
    #     path = self.find_path_to_target(active_edges, self.name_to_vertex["start"])
    #     sets = [v.name() for v in path]
    #     vertex_values = np.vstack([self.solution.GetSolution(v.x()) for v in path])
    #     return sets, vertex_values
