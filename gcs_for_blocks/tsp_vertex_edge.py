import typing as T

import numpy as np
import numpy.typing as npt


class Vertex:
    def __init__(self, name: str, value: npt.NDArray = None, block_index: int = None):
        self.value = value  # tsp-specific: value of the location
        self.name = name  # name of the vertex
        self.edges_in = []  # str names of edges in
        self.edges_out = []  # str names of edges out
        self.block_index = block_index

        self.v = None  # tsp-specific: visitation variable
        self.order = None  # tsp-specific: order variable

        self.obstacles = (
            None  # motion planning-specific: obstacles that set correponds to
        )

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

    def set_block_index(self, block_index: int):
        assert self.block_index is None, (
            "Block index for " + self.name + " is already set"
        )
        self.block_index = block_index

    def add_edge_in(self, nbh: str):
        assert nbh not in self.edges_in
        self.edges_in.append(nbh)

    def add_edge_out(self, nbh: str):
        assert nbh not in self.edges_out
        self.edges_out.append(nbh)

    def set_v(self, v):
        assert self.v is None, "V for " + self.name + " is already set"
        self.v = v

    def set_order(self, order):
        assert self.order is None, "Order for " + self.name + " is already set"
        self.order = order


class Edge:
    def __init__(
        self, left_vertex: Vertex, right_vertex: Vertex, name: str, cost: float = None
    ):
        self.left = left_vertex
        self.right = right_vertex
        self.name = name  # edge name
        self.cost = cost  # tsp-specific: float, cost over the edge

        self.phi = 0  # flow variable
        self.left_pos = 0  # mp-specific, left position
        self.right_pos = 0  # mp-specific, right position,

        self.left_order = 0  # tsp-specific, left order
        self.right_order = 0  # tsp-specific, right order

        self.left_v = 0  # tsp-specific, left visitation
        self.right_v = 0  # tsp-specific, right visitation

    def set_cost(self, cost: float):
        assert self.cost is None, "Cost for " + self.name + " is already set"
        self.cost = cost

    def set_phi(self, flow):
        assert self.phi == 0, "Flow for " + self.name + " is already set"
        self.phi = flow

    def set_left_pos(self, left_pos):
        assert self.left_pos == 0, "left_pos for " + self.name + " is already set"
        self.left_pos = left_pos

    def set_right_pos(self, right_pos):
        assert self.right_pos == 0, "right_pos for " + self.name + " is already set"
        self.right_pos = right_pos

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
