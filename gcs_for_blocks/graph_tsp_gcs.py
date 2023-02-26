import typing as T

import numpy as np
import numpy.typing as npt

from vertex import VertexTSP
from edge import EdgeTSP, Edge, EdgeGCS
from axis_aligned_set import AlignedSet

from pydrake.math import le, eq  # pylint: disable=import-error, no-name-in-module

from util import timeit, INFO, WARN, ERROR, YAY  # INFO


class ProgramOptionsForGCSTSP:
    def __init__(self):
        self.convex_relaxation_for_gcs_edges = True
        self.add_tsp_edge_costs = True
        self.add_L2_norm_cost = False

        self.solve_for_feasibility = True

        # self.objects_are_hard_constraints = True
        # self.objects_are_soft_constraints = not self.objects_are_hard_constraints

        


class GraphTSPGCS:
    def __init__(self):
        self.vertices = dict()  # type: VertexTSP
        self.edges = dict()  # type: EdgeTSP

    def s(self, name: int) -> str:
        """Name a start-block vertex"""
        return "s" + str(name) + "_tsp"

    def t(self, name: int) -> str:
        """Name a target-block vertex"""
        return "t" + str(name) + "_tsp"

    def add_tsp_vertex(
        self, name: str, block_position: npt.NDArray, block_index: int
    ) -> None:
        """Add TSP vertex to the dictionary"""
        assert name not in self.vertices, "Vertex with name " + name + " already exists"
        self.vertices[name] = VertexTSP(name, block_position, block_index)

    def add_tsp_edge(self, left_name: str, right_name: str):
        edge_name = left_name + "_" + right_name
        assert edge_name not in self.edges, "TSP Edge " + edge_name + " already exists"
        self.edges[edge_name] = EdgeTSP(
            self.vertices[left_name], self.vertices[right_name], edge_name
        )
        self.vertices[left_name].add_edge_out(edge_name)
        self.vertices[right_name].add_edge_in(edge_name)

    def add_gcs_edge(self, left_name: str, right_name: str):
        edge_name = left_name + "_" + right_name
        assert edge_name not in self.edges, "GCS Edge " + edge_name + " already exists"

        self.edges[edge_name] = EdgeGCS(
            self.vertices[left_name], self.vertices[right_name], edge_name
        )
        self.vertices[left_name].add_edge_out(edge_name)
        self.vertices[right_name].add_edge_in(edge_name)
