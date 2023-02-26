import typing as T
import numpy.typing as npt

from vertex import Vertex, VertexAlignedSet, VertexTSP, VertexTSPprogram


class Edge:
    """A simple parent Edge class"""

    def __init__(self, left_vertex: Vertex, right_vertex: Vertex, name: str) -> None:
        # TODO: should left / right be strings -- a name?
        self.left = left_vertex  # type: Vertex
        self.right = right_vertex  # type: Vertex
        self.name = name  # type: str


class EdgeGCS(Edge):
    def __init__(
        self, left_vertex: VertexTSP, right_vertex: VertexTSP, name: str
    ) -> None:
        self.left = left_vertex  # type: VertexTSP
        self.right = right_vertex  # type: VertexTSP
        self.name = name  # type: str


class EdgeMotionPlanningProgam(Edge):
    def __init__(
        self, left_vertex: VertexAlignedSet, right_vertex: VertexAlignedSet, name: str
    ):
        self.left = left_vertex
        self.right = right_vertex
        self.name = name  # edge name

        self.phi = None  # flow variable
        self.l_pos = None  # mp-specific, left position
        self.r_pos = None  # mp-specific, right position,

    def set_phi(self, flow):
        assert self.phi is None, "Flow for " + self.name + " is already set"
        self.phi = flow

    def set_l_pos(self, l_pos):
        assert self.l_pos is None, "l_pos for " + self.name + " is already set"
        self.l_pos = l_pos

    def set_r_pos(self, r_pos):
        assert self.r_pos is None, "r_pos for " + self.name + " is already set"
        self.r_pos = r_pos


class EdgeTSP(Edge):
    """A simple parent Edge class"""

    def __init__(
        self, left_vertex: VertexTSP, right_vertex: VertexTSP, name: str
    ) -> None:
        self.left = left_vertex  # type: VertexTSP
        self.right = right_vertex  # type: VertexTSP
        self.name = name  # type: str


class EdgeTSPprogram(Edge):
    def __init__(
        self,
        left_vertex: VertexTSPprogram,
        right_vertex: VertexTSPprogram,
        name: str,
        cost: float = None,
    ):
        self.left = left_vertex  # type: VertexTSPprogram
        self.right = right_vertex  # type: VertexTSPprogram
        self.name = name  # type: str
        self.cost = cost  # type: float

        self.phi = None  # flow variable
        self.left_order = None  # tsp-specific, left order
        self.right_order = None  # tsp-specific, right order
        self.left_v = None  # tsp-specific, left visitation
        self.right_v = None  # tsp-specific, right visitation

    def set_cost(self, cost: float):
        assert self.cost is None, "Cost for " + self.name + " is already set"
        self.cost = cost

    def set_phi(self, flow):
        assert self.phi is None, "Flow for " + self.name + " is already set"
        self.phi = flow

    def set_left_order(self, left_order):
        assert self.left_order is None, "left_order for " + self.name + " is already set"
        self.left_order = left_order

    def set_right_order(self, right_order):
        assert self.right_order is None, "right_order for " + self.name + " is already set"
        self.right_order = right_order

    def set_left_v(self, left_v):
        assert self.left_v is None, "left_v for " + self.name + " is already set"
        self.left_v = left_v

    def set_right_v(self, right_v):
        assert self.right_v is None, "right_v for " + self.name + " is already set"
        self.right_v = right_v
