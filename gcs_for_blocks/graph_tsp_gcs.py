import typing as T

import numpy as np
import numpy.typing as npt

from graph import Vertex, Edge, Graph
from axis_aligned_set import AlignedSet
from axis_aligned_set_tesselation import AxisAlignedSetTessellation

from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module
    MathematicalProgram,
    Solve,
)
from pydrake.math import le, eq  # pylint: disable=import-error, no-name-in-module

from .util import timeit, INFO, WARN, ERROR, YAY  # INFO


class ProgramOptionsForGCSTSP:
    def __init__(self):
        self.convex_relaxation_for_gcs_edges = True

        self.objects_are_hard_constraints = True
        self.objects_are_soft_constraints = not self.objects_are_hard_constraints

        self.add_L2_norm_cost = False


class VertexTSP(Vertex):
    def __init__(self, name:str, block_position:npt.NDArray, possible_object_index:int):
        self.name = name  # type: str
        self.edges_in = []  # type: T.List[str]
        self.edges_out = []  # type: T.List[str]

        self.block_position = block_position # type: npt.NDArray
        self.possible_object_index = possible_object_index # type: int

class EdgeTSP(Edge):
    """ A simple parent Edge class """
    def __init__(
        self, left_vertex: VertexTSP, right_vertex: VertexTSP, name: str, cost: float = None
    )->None:
        self.left = left_vertex # type: Vertex
        self.right = right_vertex # type: Vertex
        self.name = name  # type: str

        self.cost = cost # type: float

    def set_cost(self, cost: float)->None:
        assert self.cost is None, "cost being set again"
        self.cost = cost
        
class VertexTSPprogram(VertexTSP):
    def __init__(self, name, block_position, possible_object_index):
        self.name = name  # type: str
        self.edges_in = []  # type: T.List[str]
        self.edges_out = []  # type: T.List[str]
        self.block_position = block_position # type: npt.NDArray
        self.possible_object_index = possible_object_index

        self.v = None  # visitation variable: whether object-location i is occupied or not
        self.order = None  # tsp-specific: order variable

    @staticmethod
    def from_vertex_tsp(vertex: VertexTSP) -> "VertexTSPprogram":
        return VertexTSPprogram(vertex.name, vertex.block_position, vertex.possible_object_index)

    def set_v(self, v):
        assert self.v is None, "V for " + self.name + " is already set"
        self.v = v

    def set_order(self, order):
        assert self.order is None, "Order for " + self.name + " is already set"
        self.order = order

class EdgeTSPprogram(EdgeTSP):
    def __init__(
        self, left_vertex: VertexTSPprogram, right_vertex: VertexTSPprogram, name: str, cost: float = None
    ):
        self.left = left_vertex # type: VertexTSPprogram
        self.right = right_vertex # type: VertexTSPprogram
        self.name = name  # type: str
        self.cost = cost # type: float

        self.phi = None  # flow variable
        self.left_order = None  # tsp-specific, left order
        self.right_order = None  # tsp-specific, right order
        self.left_v = None  # tsp-specific, left visitation
        self.right_v = None  # tsp-specific, right visitation

    def set_cost(self, cost: float):
        assert self.cost is None, "Cost for " + self.name + " is already set"
        self.cost = cost

    def set_phi(self, flow):
        assert self.phi == 0, "Flow for " + self.name + " is already set"
        self.phi = flow

    def set_left_order(self, left_order):
        assert self.left_order == 0, "left_order for " + self.name + " is already set"
        self.left_order = left_order

    def set_right_order(self, right_order):
        assert self.right_order == 0, "right_order for " + self.name + " is already set"
        self.right_order = right_order

    def set_left_v(self, left_v):
        assert self.left_v == 0, "left_v for " + self.name + " is already set"
        self.left_v = left_v

    def set_right_v(self, right_v):
        assert self.right_v == 0, "right_v for " + self.name + " is already set"
        self.right_v = right_v

class GraphTSPGCS:
    def __init__(self):
        self.vertices = dict() # type: VertexTSP
        self.edges = dict() # type: EdgeTSP

    def s(self, name: int) -> str:
        """Name a start-block vertex"""
        return "s" + str(name) + "_tsp"

    def t(self, name: int) -> str:
        """Name a target-block vertex"""
        return "t" + str(name) + "_tsp"

    def add_tsp_vertex(self, name: str, value: npt.NDArray, block_index: int) -> None:
        """Add TSP vertex to the dictionary"""
        assert name not in self.vertices, "Vertex with name " + name + " already exists"
        self.vertices[name] = VertexTSP(name, value, block_index)

    def add_tsp_edge(self, left_name: str, right_name:str):
        edge_name = left_name + "_" + right_name
        assert edge_name not in self.edges, "Edge " + edge_name + " already exists"
        self.edges[edge_name] = Edge(self.vertices[left_name], self.vertices[right_name], edge_name)
        self.vertices[left_name].add_edge_out(edge_name)
        self.vertices[right_name].add_edge_in(edge_name)

    def add_gcs_edge(self, left_name:str, right_name:str):


    # add a GCS edge
        # for now assume that any GCS edge connects just two vertices

    # trasncribe problem
        # params: 
        # optimal -- use L2 norms for motion planning
        # relaxed, objects are hard constraints
        # relaxed, objects are large costs
        # return solution
        # get order from solution
        # go down the gradient
        





def construct_tsp_gcs(obstacles: T.List[AlignedSet], start_objects: T.List[AlignedSet], target_objects: T.List[AlignedSet], arm_start_pos, arm_target_pos):
    # make a tessellation
    tessellation = None

    graph = GraphTSPGCS()

    possible_object_locations = start_objects + target_objects

    # add the vertices
    for i, obj in enumerate(start_objects):
        graph.add_tsp_vertex(graph.s(i), obj.center, i) 
    for i, obj in enumerate(target_objects):
        graph.add_tsp_vertex(graph.t(i), obj.center, i + len(start_objects))

    # add arm start/target TSP vertices
    # you can just call them start and target honestly
    graph.add_tsp_vertex("start", arm_start_pos, -1)
    graph.add_tsp_vertex("target", arm_target_pos, -1)


    # add TSP edges
    for i in range(len(start_objects)):
        # arm start to any start
        graph.add_tsp_edge("start", graph.s(i))
        # target any to target
        graph.add_tsp_edge(graph.t(i), "target")
        # target any to start any 
        for j in range(len(start_objects)):
            if i != j:
                graph.add_tsp_edge(graph.t(i), graph.s(j))
    


    