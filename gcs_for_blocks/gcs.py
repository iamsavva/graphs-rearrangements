# pyright: reportMissingImports=false
import typing as T

import numpy as np
import numpy.typing as npt

import pydot
from tqdm import tqdm
from IPython.display import Image, display
import time

# from PIL import Image as PIL_Image

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

from .util import ERROR, WARN, INFO, YAY, timeit
from .gcs_options import GCSforBlocksOptions, EdgeOptions
from .gcs_set_generator import GCSsetGenerator


class GCSforBlocks:
    """
    GCS for N-dimensional block moving using a top-down suction cup.
    """

    ###################################################################################
    # Properties, inits, setter functions

    def __init__(self, options: GCSforBlocksOptions):
        # options
        self.opt = options

        # init the graph
        self.gcs = GraphOfConvexSets()
        self.solution = None
        self.graph_built = False

        self.set_gen = GCSsetGenerator(options)

        # the following structures hold information about the graph connectivity.

        # name to vertex dictionary, populated as we populate the graph with vertices
        self.name_to_vertex = dict()  # T.Dict[str, GraphOfConvexSets.Vertex]

        # below structures are used for hand-built graph edge information.
        # see populate_important_things()
        # set ids located in a mode
        self.sets_per_mode = dict()  # T.Dict[int, T.Set[int]]
        # modes that are represented in a layer (horizon step)
        self.modes_per_layer = dict()  # T.Dict[int, T.Set[int]]
        # set_ids that are represented ina  layer (horizon step)
        self.sets_per_layer = dict()  # T.Dict[int, T.Set[int]]
        # edges matrix for a mode graph
        self.mode_graph_edges = np.empty([])  # np.NDArray, size num_modes x num_modes
        # edge matrix for a set graph
        self.set_graph_edges = np.array(
            []
        )  # np.NDArray, size num_gcs_sets x num_gcs_sets
        # get polyhedron that describes a set
        self.set_id_to_polyhedron = dict()  # T.Dict[int, HPolyhedron]
        self.opt.num_gcs_sets = -1  # int, numer of GCS sets.

    ###################################################################################
    # Building the finite horizon GCS

    def build_the_graph(
        self,
        start_state: Point,
        start_mode: int,
        target_state: Point,
        target_mode: int,
    ) -> None:
        """
        Build the GCS graph of horizon H from start to target nodes.
        TODO:
        - allow target state to be a set
        """
        # reset the graph
        self.gcs = GraphOfConvexSets()
        self.graph_built = False
        self.start_mode = start_mode
        self.target_mode = target_mode

        # hand-built pre-processing of the graph
        self.populate_important_things()

        # add all vertices
        self.add_all_vertices(start_state, target_state)
        # add all edges
        self.add_all_edges()

        self.graph_built = True

    ###################################################################################
    # Adding layers of nodes (trellis diagram style)

    def add_all_vertices(
        self,
        start_state: Point,
        target_state: Point,
    ) -> None:
        # add start node to the graph
        self.add_vertex(start_state, "start")
        # add vertices into horizon 0
        for set_id in self.sets_per_mode[self.start_mode]:
            self.add_vertex(
                self.get_convex_set_for_set_id(set_id), self.get_vertex_name(0, set_id)
            )
        # add vertices into horizon 1 through last
        for layer in range(1, self.opt.horizon):
            for mode in self.modes_per_layer[layer]:
                # for each set in that mode, add new vertex
                for set_id in self.sets_per_mode[mode]:
                    vertex_name = self.get_vertex_name(layer, set_id)
                    convex_set = self.get_convex_set_for_set_id(set_id)
                    self.add_vertex(convex_set, vertex_name)
        # add target vertex
        self.add_vertex(target_state, "target")

    def add_all_edges(self) -> None:
        ############################
        # between start and layer 0

        # get sets that intersect with the start set
        sets_with_start = self.get_sets_in_mode_that_intersect_with_set(
            self.start_mode,
            self.name_to_vertex["start"].set(),
            just_one=self.opt.connect_source_target_to_single_set,
        )
        names_of_sets_with_start = self.set_names_for_layer(sets_with_start, 0)
        self.connect_to_vertex_on_the_right(
            "start", names_of_sets_with_start, EdgeOptions.equality_edge()
        )

        ############################
        # edges within layers and between layers
        for layer in range(self.opt.horizon):
            for mode in self.modes_per_layer[layer]:
                # now add edges
                # TODO: this should be done more carefully in the future (?)
                for set_id in self.sets_per_mode[mode]:
                    vertex_name = self.get_vertex_name(layer, set_id)

                    # add edges into vertex from the previous layer
                    if layer > 0:
                        edges_in = self.get_edges_into_set_out_of_mode(set_id)
                        names_of_edges_in = self.set_names_for_layer(
                            edges_in, layer - 1
                        )
                        self.connect_to_vertex_on_the_left(
                            names_of_edges_in,
                            vertex_name,
                            EdgeOptions.between_modes_edge(self.opt.add_grasp_cost),
                        )

                    # edges out of vertex into the same mode of same layer
                    intra_mode_in = self.get_edges_within_same_mode(set_id)
                    names_of_intra_mode = self.set_names_for_layer(intra_mode_in, layer)
                    self.connect_to_vertex_on_the_right(
                        vertex_name, names_of_intra_mode, EdgeOptions.within_mode_edge()
                    )

        ##############################
        # edges to target

        # sets in target_mode that intersect with target_state
        sets_with_target = self.get_sets_in_mode_that_intersect_with_set(
            self.target_mode,
            self.name_to_vertex["target"].set(),
            just_one=self.opt.connect_source_target_to_single_set,
        )
        names_of_sets_with_target = []
        # at each horizon level, only sets that contain the target can transition into target
        # for layer in range(self.opt.horizon):
        for layer in (self.opt.horizon - 1,):
            # if that layer has a target mode
            if self.target_mode in self.modes_per_layer[layer]:
                # for each set that contains the target
                for set_id in sets_with_target:
                    names_of_sets_with_target += [self.get_vertex_name(layer, set_id)]
        # add the edges
        self.connect_to_vertex_on_the_left(
            names_of_sets_with_target, "target", EdgeOptions.within_mode_edge()
        )

    ###################################################################################
    # Populating edges and vertices

    def add_edge(
        self,
        left_vertex: GraphOfConvexSets.Vertex,
        right_vertex: GraphOfConvexSets.Vertex,
        edge_opt: EdgeOptions,
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
        # add an orbital constraint
        if edge_opt.add_orbital_constraint:
            left_mode = self.get_mode_from_vertex_name(left_vertex.name())
            self.add_orbital_constraint(left_mode, edge)
        if edge_opt.add_set_transition_constraint:
            left_set_id = self.get_set_id_from_vertex_name(left_vertex.name())
            self.add_common_set_at_transition_constraint(left_set_id, edge)
        if edge_opt.add_equality_constraint:
            self.add_point_equality_constraint(edge)
        # -----------------------------------------------------------------
        # Adding costs
        # -----------------------------------------------------------------
        # add movement cost on the edge
        if edge_opt.add_gripper_movement_cost:
            self.add_gripper_movement_cost_on_edge(edge)
        # add time cost on edge
        if edge_opt.add_grasp_cost:
            self.add_grasp_cost_on_edge(edge)

    def add_vertex(self, convex_set: HPolyhedron, name: str) -> None:
        """
        Define a vertex with a convex set.
        """
        # print(self.name_to_vertex.keys())
        # print(name)
        if name not in self.name_to_vertex:
            # print("adding " + name)
            vertex = self.gcs.AddVertex(convex_set, name)
            self.name_to_vertex[name] = vertex
            # if not convex_set.IsBounded():
            #     WARN("Convex set for", name, "is not bounded!")
            # return vertex
        # else:
        #     return self.name_to_vertex[name]

    def connect_vertices(
        self, left_vertex_name: str, right_vertex_name: str, edge_opt: EdgeOptions
    ):
        """Connect vertices by name"""
        left_vertex = self.name_to_vertex[left_vertex_name]
        right_vertex = self.name_to_vertex[right_vertex_name]

        self.add_edge(left_vertex, right_vertex, edge_opt)

    def connect_to_vertex_on_the_right(
        self,
        left_vertex_name: str,
        right_vertex_names: T.List[str],
        edge_opt: EdgeOptions,
    ):
        """Left vertex gets connected to set of right vertices."""
        for right_vertex_name in right_vertex_names:
            self.connect_vertices(left_vertex_name, right_vertex_name, edge_opt)

    def connect_to_vertex_on_the_left(
        self,
        left_vertex_names: T.List[str],
        right_vertex_name: str,
        edge_opt: EdgeOptions,
    ):
        """Set of left vertices get connected to right vertex."""
        for left_vertex_name in left_vertex_names:
            self.connect_vertices(left_vertex_name, right_vertex_name, edge_opt)

    ###################################################################################
    # Adding constraints and cost terms
    def add_orbital_constraint(
        self, left_mode: int, edge: GraphOfConvexSets.Edge
    ) -> None:
        """
        READY
        Add orbital constraints on the edge
        Orbital constraints are independent of edges
        """
        xu, xv = edge.xu(), edge.xv()
        orbital_constraint = self.set_gen.get_orbital_constraint(left_mode)
        edge.AddConstraint(
            Binding[LinearEqualityConstraint](orbital_constraint, np.append(xv, xu))
        )

    def add_common_set_at_transition_constraint(
        self, left_vertex_set_id: int, edge: GraphOfConvexSets.Edge
    ) -> None:
        """
        READY
        Add a constraint that the right vertex belongs to the same mode as the left vertex
        """
        # get set that corresponds to left vertex
        left_vertex_set = self.get_convex_set_for_set_id(left_vertex_set_id)
        # fill in linear constraint on the right vertex
        A = left_vertex_set.A()
        lb = -np.ones(left_vertex_set.b().size) * 1000
        ub = left_vertex_set.b()
        set_con = LinearConstraint(A, lb, ub)
        edge.AddConstraint(Binding[LinearConstraint](set_con, edge.xv()))

    def add_point_equality_constraint(self, edge: GraphOfConvexSets.Edge) -> None:
        """
        READY
        Add a constraint that the right vertex belongs to the same mode as the left vertex
        """
        # get set that corresponds to left vertex
        A = np.hstack((np.eye(self.opt.state_dim), -np.eye(self.opt.state_dim)))
        b = np.zeros(self.opt.state_dim)
        set_con = LinearEqualityConstraint(A, b)
        edge.AddConstraint(
            Binding[LinearEqualityConstraint](set_con, np.append(edge.xu(), edge.xv()))
        )

    def add_gripper_movement_cost_on_edge(self, edge: GraphOfConvexSets.Edge) -> None:
        """
        READY
        L2 norm cost on the movement of the gripper.
        """
        xu, xv = edge.xu(), edge.xv()
        #  gripper state is 0 to block_dim
        d = self.opt.block_dim
        n = self.opt.state_dim
        A = np.zeros((d, 2 * n))
        A[:, 0:d] = np.eye(d)
        A[:, n : n + d] = -np.eye(d)
        b = np.zeros(d)
        # add the cost
        cost = L2NormCost(A, b)
        edge.AddCost(Binding[L2NormCost](cost, np.append(xv, xu)))

    def add_grasp_cost_on_edge(self, edge: GraphOfConvexSets.Edge) -> None:
        """
        READY
        Walking along the edges costs some cosntant term. This is done to avoid grasping and ungrasping in place.
        """
        A = np.zeros(self.opt.state_dim)
        b = self.opt.time_cost_weight * np.ones(1)
        cost = LinearCost(A, b)
        edge.AddCost(Binding[LinearCost](cost, edge.xv()))

    ###################################################################################
    # build sets that are inside the modes

    def build_sets_per_mode(self) -> None:
        """
        Determine which sets belong to which mode
        """
        if self.opt.problem_complexity == "transparent-no-obstacles":
            # each mode set is just the mode itself
            for mode in range(self.opt.num_modes):
                self.sets_per_mode[mode] = {mode}
            # number of gcs sets is the number of modes
            self.opt.num_gcs_sets = self.opt.num_modes

        elif self.opt.problem_complexity == "obstacles":
            # mode 0 is collision free and hence has just a single set in it
            # TODO: this is not the case if i have obstacles
            self.sets_per_mode[0] = {0}
            self.set_id_to_polyhedron[
                0
            ] = self.set_gen.get_convex_set_for_mode_polyhedron(0)
            set_indexer = 1
            # for each mode
            for mode in range(1, self.opt.num_modes):
                # get convex sets that belong to this mode
                sets_in_mode = self.set_gen.get_convex_tesselation_for_mode(mode)
                self.sets_per_mode[mode] = set()
                INFO("mode ", mode, "has convex sets:", len(sets_in_mode))
                # for each IRIS set: redice inequalities, add the set to the dictionaries
                for some_set in sets_in_mode:
                    some_set = some_set.ReduceInequalities()
                    self.sets_per_mode[mode].add(set_indexer)
                    self.set_id_to_polyhedron[set_indexer] = some_set
                    set_indexer += 1
            self.opt.num_gcs_sets = set_indexer

    def get_mode_from_set_id(self, set_id: int) -> int:
        """
        READY
        Returns a mode to which the vertex belongs.
        In the simple case of transparent blocks, it's just the vertex (the mode) itself.
        """
        assert 0 <= set_id < self.opt.num_gcs_sets, "Set number out of bounds"
        if self.opt.problem_complexity == "transparent-no-obstacles":
            # eahc mode is the set itself
            mode = set_id
            return mode

        elif self.opt.problem_complexity == "obstacles":
            # must find a mode that has the set
            for mode in range(self.opt.num_modes):
                if set_id in self.sets_per_mode[mode]:
                    return mode
            assert False, "Set_id not in any mode??"
        raise NotImplementedError

    def get_convex_set_for_set_id(self, set_id: int) -> HPolyhedron:
        """
        READY
        Returns convex set that corresponds to the given vertex.
        For the simple case of transparent blocks, it's the convex set that corresponds to the mode.
        """
        assert self.opt.problem_complexity in (
            "transparent-no-obstacles",
            "obstacles",
        ), "Problem complexity option not implemented"
        if self.opt.problem_complexity == "transparent-no-obstacles":
            mode = set_id
            return self.set_gen.get_convex_set_for_mode_polyhedron(mode)
        elif self.opt.problem_complexity == "obstacles":
            return self.set_id_to_polyhedron[set_id]
        raise NotImplementedError

    def get_sets_in_mode_that_intersect_with_set(
        self, mode: int, my_set: ConvexSet, just_one: bool = False
    ) -> T.List[HPolyhedron]:
        """
        Get all convex sets in mode that intersect with my_set. Or give me just one.
        Returns a list of polyhedra.
        """
        sets_in_mode = self.sets_per_mode[mode]
        sets_with_my_set = []
        for set_in_mode in sets_in_mode:
            convex_set = self.get_convex_set_for_set_id(set_in_mode)
            if convex_set.IntersectsWith(my_set):
                sets_with_my_set += [set_in_mode]
                if just_one:
                    break
        assert (
            len(sets_with_my_set) > 0
        ), "No set in given mode intersect with the given set!"
        return sets_with_my_set

    ###################################################################################
    # Functions related to connectivity between modes and sets within modes

    def populate_important_things(self) -> None:
        """
        READY
        Preprocessing step that populates various graph-related things in an appropriate order.
        """
        # Run IRIS to determine sets that lie in individual modes
        self.build_sets_per_mode()
        # determine mode connectivity
        self.populate_edges_between_modes()
        # determine set connectivity
        self.populate_edges_between_sets()
        # determine which modes appear in which layer
        self.populate_modes_per_layer(self.start_mode)
        # determine which sets appear in which layer
        self.populate_sets_per_layer()

    def populate_edges_between_modes(self) -> None:
        """
        READY
        Mode connectivity.
        """
        self.mode_graph_edges = np.zeros((self.opt.num_modes, self.opt.num_modes))
        if self.opt.allow_self_transitions_for_modes:
            # mode 0 is connected to any other mode and itself
            self.mode_graph_edges[0, :] = np.ones(self.opt.num_modes)
            # mode k is connected only to 0 and itself
            self.mode_graph_edges[:, 0] = np.ones(self.opt.num_modes)
            for i in range(1, self.opt.num_modes):
                self.mode_graph_edges[i, i] = 1
        else:
            # mode 0 is connected to any other mode except itself
            self.mode_graph_edges[0, 1:] = np.ones(self.opt.num_modes - 1)
            # mode k is connected only to 0;
            self.mode_graph_edges[1:, 0] = np.ones(self.opt.num_modes - 1)

    def populate_edges_between_sets(self) -> None:
        """
        READY
        Return a matrix that represents edges in a directed graph of modes.
        For this simple example, the matrix is hand-built.
        When IRIS is used, sets must be A -- clustered (TODO: do they?),
        B -- connectivity checked and defined automatically.
        """
        self.set_graph_edges = np.zeros((self.opt.num_gcs_sets, self.opt.num_gcs_sets))
        if self.opt.problem_complexity == "transparent-no-obstacles":
            # mode 0 is connected to any other mode except itself
            self.set_graph_edges[0, 1:] = np.ones(self.opt.num_gcs_sets - 1)
            # mode k is connected only to 0;
            self.set_graph_edges[1:, 0] = np.ones(self.opt.num_gcs_sets - 1)

        elif self.opt.problem_complexity == "obstacles":
            # TODO: this a slight hack, technically needs better implementation
            # TODO: specifically, this is because i directly use the knowledge that 0 to
            # everything everythin to 0

            # TODO: edges ought to be populated manually
            # for each set in mode 0, connnect 0 to other modes
            for i in self.sets_per_mode[0]:
                poly_i = self.set_id_to_polyhedron[i]
                # for each other set
                for j in range(i + 1, self.opt.num_gcs_sets):
                    poly_j = self.set_id_to_polyhedron[j]
                    # if they intersect -- add an edge
                    if poly_i.IntersectsWith(poly_j):
                        self.set_graph_edges[i, j] = 1
                        self.set_graph_edges[j, i] = 1

            # connect other modes with themselves
            for mode in range(1, self.opt.num_modes):
                sets_in_mode = self.sets_per_mode[mode]
                for set_id in sets_in_mode:
                    convex_set = self.get_convex_set_for_set_id(set_id)
                    for other_set_id in sets_in_mode:
                        if set_id != other_set_id:
                            other_convex_set = self.get_convex_set_for_set_id(
                                other_set_id
                            )
                            if convex_set.IntersectsWith(other_convex_set):
                                self.set_graph_edges[set_id, other_set_id] = 1
                                self.set_graph_edges[other_set_id, set_id] = 1

    def populate_modes_per_layer(self, start_mode: int) -> None:
        """
        READY
        For each layer, determine what modes are reachable from the start node.
        """
        # at horizon 0 we have just the start mode
        self.modes_per_layer[0] = {start_mode}
        # for horizons 1 through h-1:
        for h in range(1, self.opt.horizon):
            modes_at_next_layer = set()
            # for each modes at previous horizon
            for m in self.modes_per_layer[h - 1]:
                # add anything connected to it
                for k in self.get_edges_out_of_mode(m):
                    modes_at_next_layer.add(k)
            self.modes_per_layer[h] = modes_at_next_layer

    def populate_sets_per_layer(self) -> None:
        """
        READY
        If a mode belongs to a layer, all of its sets belong to a layer.
        """
        assert (
            len(self.modes_per_layer) == self.opt.horizon
        ), "Must populate modes per layer first"
        # for each layer up to the horizon
        for layer in range(self.opt.horizon):
            self.sets_per_layer[layer] = set()
            # for each mode in that layer
            for mode_in_layer in self.modes_per_layer[layer]:
                # for each set in that mode
                for set_in_mode in self.sets_per_mode[mode_in_layer]:
                    # add that set to the (set of sets) at that layer
                    self.sets_per_layer[layer].add(set_in_mode)

    ###################################################################################
    # Get edges in and out of a set

    def get_edges_into_set_out_of_mode(self, set_id: int) -> T.List[int]:
        """
        READY
        Use the edge matrix to determine which edges go into the vertex.
        """
        assert 0 <= set_id < self.opt.num_gcs_sets, "Set number out of bounds"
        if self.opt.problem_complexity == "transparent-no-obstacles":
            return [
                v
                for v in range(self.opt.num_gcs_sets)
                if self.set_graph_edges[v, set_id] == 1
            ]
        elif self.opt.problem_complexity == "obstacles":
            edges = []
            mode = self.get_mode_from_set_id(set_id)
            # for each mode that enters into our mode
            modes_into_me = self.get_edges_into_mode(mode)
            for other_mode in modes_into_me:
                assert (
                    other_mode != mode
                ), "I am not supposed to get same-mode edges from out of mode into me"
                # for each set in that other mode
                for i in self.sets_per_mode[other_mode]:
                    # check if there is an edge
                    if self.set_graph_edges[i, set_id] == 1:
                        edges.append(i)
            return edges
        raise NotImplementedError

    def get_edges_within_same_mode(self, set_id: int) -> T.List[int]:
        """
        READY
        edge matrix should contain only values for out of mode transitions
        here is where we deal with within-the-mode transitions
        """
        assert 0 <= set_id < self.opt.num_gcs_sets, "Set number out of bounds"
        if self.opt.problem_complexity == "transparent-no-obstacles":
            return []
        elif self.opt.problem_complexity == "obstacles":
            mode = self.get_mode_from_set_id(set_id)
            edges = []
            for i in self.sets_per_mode[mode]:
                if i != set_id and self.set_graph_edges[set_id, i] == 1:
                    edges.append(i)
            return edges
        raise NotImplementedError

    def get_edges_out_of_set_out_of_mode(self, set_id: int) -> T.List[int]:
        """
        READY
        Use the edge matrix to determine which edges go out of the vertex.
        """
        assert 0 <= set_id < self.opt.num_gcs_sets, "Set number out of bounds"
        if self.opt.problem_complexity == "transparent-no-obstacles":
            return [
                v
                for v in range(self.opt.num_gcs_sets)
                if self.set_graph_edges[set_id, v] == 1
            ]
        elif self.opt.problem_complexity == "obstacles":
            edges = []
            mode = self.get_mode_from_set_id(set_id)
            # for each mode that we go into
            modes_out_of_me = self.get_edges_out_of_mode(mode)
            for other_mode in modes_out_of_me:
                assert other_mode != mode
                # for each set in that other mode
                for i in self.sets_per_mode[other_mode]:
                    # check if there is an edge
                    if self.set_graph_edges[set_id, i] == 1:
                        edges.append(i)
            return edges
        raise NotImplementedError

    def get_edges_into_set(self, set_id):
        return self.get_edges_into_set_out_of_mode(
            set_id
        ) + self.get_edges_within_same_mode(set_id)

    def get_edges_out_of_set(self, set_id):
        return self.get_edges_out_of_set_out_of_mode(
            set_id
        ) + self.get_edges_within_same_mode(set_id)

    def get_edges_out_of_mode(self, mode: int) -> T.List[int]:
        """
        READY
        Use the edge matrix to determine which edges go out of the vertex.
        """
        assert 0 <= mode < self.opt.num_modes, "Mode out of bounds"
        return [
            v for v in range(self.opt.num_modes) if self.mode_graph_edges[mode, v] == 1
        ]

    def get_edges_into_mode(self, mode: int) -> T.List[int]:
        """
        READY
        Use the edge matrix to determine which edges go out of the vertex.
        """
        assert 0 <= mode < self.opt.num_modes, "Mode out of bounds"
        return [
            v for v in range(self.opt.num_modes) if self.mode_graph_edges[v, mode] == 1
        ]

    ###################################################################################
    # Vertex and edge naming

    def get_vertex_name(self, layer: int, set_id: int, t="") -> str:
        """Naming convention is: M_<layer>_<set_id> for regular nodes"""
        return t + "M_" + str(layer) + "_" + str(set_id)

    def get_set_id_from_vertex_name(self, name: str) -> int:
        assert name not in ("start", "target"), "Trying to get set id for bad sets!"
        set_id = int(name.split("_")[-1])
        return set_id

    def get_mode_from_vertex_name(self, name: str) -> int:
        if name == "start":
            return self.start_mode
        if name == "target":
            return self.target_mode
        set_id = self.get_set_id_from_vertex_name(name)
        return self.get_mode_from_set_id(set_id)

    def get_edge_name(self, left_vertex_name: str, right_vertex_name: str) -> str:
        if right_vertex_name == "target":
            layer = int(left_vertex_name.split("_")[-2])
            return "Free move to target at " + str(layer)
        if left_vertex_name == "start":
            return "Equals start"
        layer = int(left_vertex_name.split("_")[-2])
        left_mode = self.get_mode_from_vertex_name(left_vertex_name)
        right_mode = self.get_mode_from_vertex_name(right_vertex_name)
        if left_mode in ("0", 0):
            return "Free move, grasp " + str(right_mode)  # + " at " + str(layer)
        else:
            return "Grasping move, ungrasp " + str(left_mode)  # + " at " + str(layer)
        # return "E: " + left_vertex_name + " -> " + right_vertex_name

    def set_names_for_layer(self, set_ids, layer):
        return [self.get_vertex_name(layer, set_id) for set_id in set_ids]

    ###################################################################################
    # Solve and display solution

    def solve(
        self,
        use_convex_relaxation=None,
        max_rounded_paths=None,
        show_graph=False,
        graph_name="temp",
        verbose=True,
    ):
        """Solve the GCS program. Must build the graph first."""
        assert self.graph_built, "Must build graph first!"
        if use_convex_relaxation is None:
            use_convex_relaxation = self.opt.use_convex_relaxation
        if max_rounded_paths is None:
            max_rounded_paths = self.opt.max_rounded_paths

        start_vertex = self.name_to_vertex["start"].id()
        target_vertex = self.name_to_vertex["target"].id()
        options = opt.GraphOfConvexSetsOptions()
        options.convex_relaxation = use_convex_relaxation
        options.preprocessing = True  # TODO Do I need to deal with this?
        if use_convex_relaxation:
            options.preprocessing = True  # TODO Do I need to deal with this?
            options.max_rounded_paths = max_rounded_paths
            options.rounding_seed = self.opt.rounding_seed
        INFO("Solving...", verbose=verbose)
        start = time.time()
        self.solution = self.gcs.SolveShortestPath(start_vertex, target_vertex, options)
        if self.solution.is_success():
            YAY(
                "Solving GCS took %.2f seconds" % (time.time() - start), verbose=verbose
            )
            YAY(
                "Optimal cost is %.5f" % self.solution.get_optimal_cost(),
                verbose=verbose,
            )
        else:
            ERROR("SOLVE FAILED!", verbose=verbose)
            ERROR(
                "Solving GCS took %.2f seconds" % (time.time() - start), verbose=verbose
            )
        if show_graph:
            self.display_graph(graph_name)

    def solve_plot_sparse(
        self,
        use_convex_relaxation=None,
        max_rounded_paths=None,
    ):
        self.solve(
            use_convex_relaxation=use_convex_relaxation,
            max_rounded_paths=max_rounded_paths,
            show_graph=True,
            graph_name="temp_original",
        )
        assert self.solution.is_success(), "Solution was not found"
        for e in self.gcs.Edges():
            if not 0.01 <= self.solution.GetSolution(e.phi()):
                self.gcs.RemoveEdge(e.id())
        for v in self.gcs.Vertices():
            if np.any(np.isnan(self.solution.GetSolution(v.x()))):
                self.gcs.RemoveVertex(v.id())
            if np.allclose(
                self.solution.GetSolution(v.x()), np.zeros(self.opt.state_dim)
            ):
                self.gcs.RemoveVertex(v.id())
        self.solve(
            use_convex_relaxation=use_convex_relaxation,
            max_rounded_paths=max_rounded_paths,
            show_graph=True,
            graph_name="temp_non_empty",
        )

    def display_graph(self, graph_name="temp") -> None:
        """Visually inspect the graph. If solution acquired -- also displays the solution."""
        assert self.graph_built, "Must build graph first!"
        if self.solution is None or not self.solution.is_success():
            graphviz = self.gcs.GetGraphvizString()
        else:
            graphviz = self.gcs.GetGraphvizString(self.solution, True, precision=2)
        data = pydot.graph_from_dot_data(graphviz)[0]  # type: ignore
        data.write_png(graph_name + ".png")
        data.write_svg(graph_name + ".svg")

        # plt = Image(data.create_png())
        # display(plt)

    def find_path_to_target(
        self,
        edges: T.List[GraphOfConvexSets.Edge],
        start: GraphOfConvexSets.Vertex,
    ) -> T.List[GraphOfConvexSets.Vertex]:
        """Given a set of active edges, find a path from start to target"""
        # seed
        np.random.seed(self.opt.rounding_seed)

        edges_out = [e for e in edges if e.u() == start]
        flows_out = np.array([self.solution.GetSolution(e.phi()) for e in edges_out])
        proabilities = np.where(flows_out < 0, 0, flows_out)
        proabilities /= sum(proabilities)

        current_edge = np.random.choice(edges_out, 1, p=proabilities)[0]
        # current_edge = np.random.choice(edges_out, 1)[0]
        # get the next vertex and continue
        v = current_edge.v()

        target_reached = v == self.name_to_vertex["target"]

        if target_reached:
            return [start] + [v], [current_edge]
        else:
            v, e = self.find_path_to_target(edges, v)
            return [start] + v, [current_edge] + e

    def get_solution_path(self) -> T.Tuple[T.List[str], npt.NDArray]:
        """Given a solved GCS problem, and assuming it's tight, find a path from start to target"""
        assert self.graph_built, "Must build graph first!"
        assert self.solution.is_success(), "Solution was not found"
        # find edges with non-zero flow
        flow_variables = [e.phi() for e in self.gcs.Edges()]
        flow_results = [self.solution.GetSolution(p) for p in flow_variables]
        not_tight = np.any(
            np.logical_and(0.05 < np.array(flow_results), np.array(flow_results) < 0.95)
        )
        if not_tight:
            WARN("Solution s not tight, returning A path, not THE optimal path")

        active_edges = [
            edge for edge, flow in zip(self.gcs.Edges(), flow_results) if flow > 0.0
        ]
        if not_tight:
            # gen random paths
            vertex_paths, edge_paths = [], []
            for i in range(max(1, self.opt.custom_rounding_paths)):
                vertices, edges = self.find_path_to_target(
                    active_edges, self.name_to_vertex["start"]
                )
                vertex_paths += [vertices]
                edge_paths += [edges]
                self.opt.rounding_seed += 1
            # resolve
            solutions = []
            for vertices, edges in zip(vertex_paths, edge_paths):
                for e in self.gcs.Edges():
                    if e not in edges:
                        e.AddPhiConstraint(False)
                self.solve(
                    use_convex_relaxation=False, max_rounded_paths=0, verbose=False
                )
                active_edges = [
                    edge
                    for edge, flow in zip(self.gcs.Edges(), flow_results)
                    if flow > 0.0
                ]
                v_path, _ = self.find_path_to_target(
                    active_edges, self.name_to_vertex["start"]
                )
                v_name_path = [v.name() for v in v_path]
                cost = self.solution.get_optimal_cost()
                self.gcs.ClearAllPhiConstraints()
                solutions.append((v_name_path, cost))

            costs = np.array([cost for (_, cost) in solutions])
            print(
                "Min:",
                np.min(costs),
                "\nMax:",
                np.max(costs),
                "\nAverage:",
                np.mean(costs),
                "\nSTD:",
                np.std(costs),
            )
            return solutions

        # using these edges, find the path from start to target
        path, _ = self.find_path_to_target(active_edges, self.name_to_vertex["start"])
        modes = [v.name() for v in path]
        if self.opt.problem_complexity != "collision-free-all-moving":
            modes = [
                str(self.get_mode_from_vertex_name(mode))
                if mode not in ("start", "target")
                else mode
                for mode in modes
            ]
        vertex_values = np.vstack([self.solution.GetSolution(v.x()) for v in path])
        return modes, vertex_values

    def verbose_solution_description(self) -> None:
        """Describe the solution in text: grasp X, move to Y, ungrasp Z"""
        assert self.solution.is_success(), "Solution was not found"
        modes, vertices = self.get_solution_path()
        for i in range(len(vertices)):
            vertices[i] = ["%.2f" % v for v in vertices[i]]
        mode_now = modes[1]
        INFO("-----------------------")
        INFO("Solution is:")
        INFO("-----------------------")
        for i in range(len(modes)):  # pylint: disable=consider-using-enumerate
            sg = vertices[i][0 : self.opt.block_dim]
            if modes[i] == "start":
                INFO("Start at", sg)
            elif modes[i] == "target":
                INFO("Move to", sg, "; Finish")
            else:
                mode_next = modes[i]
                if mode_next == mode_now:
                    grasp = ""
                elif mode_next == "0":
                    grasp = "Ungrasp block " + str(mode_now)
                else:
                    grasp = "Grasp block " + str(mode_next)
                mode_now = mode_next
                INFO("Move to", sg, "; " + grasp)
