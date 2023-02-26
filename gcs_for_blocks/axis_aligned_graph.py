import typing as T

import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from vertex import Vertex, VertexAlignedSet
from edge import Edge
from axis_aligned_set import AlignedSet
from axis_aligned_set_tesselation import (
    AxisAlignedSetTessellation,
    make_a_test_with_obstacles,
    make_swap_two_test,
    make_a_test_with_objects_and_obstacles
)

class GraphOfAdjacentAlignedSets:
    def __init__(self, tessellation: AxisAlignedSetTessellation):
        self.vertices = dict()  # type: VertexAlignedSet
        self.edges = dict()  # type: Edge
        self.tessellation = tessellation
        self.make_graph_from_tessellation()

    ################################################################################
    # Making a graph + finding a set in a graph that is relevant
    # @staticmethod
    def make_graph_from_tessellation(self):
        # add all vertices
        self.tessellation.add_names_to_sets()
        for aligned_set in self.tessellation.tessellation_set:
            # do not add obstacles to the graph
            if not aligned_set.is_obstacle():
                self.add_vertex(VertexAlignedSet(aligned_set.name, aligned_set))

        # add all edges, so far as objects are nearby
        for v1 in self.vertices.values():
            for v2 in self.vertices.values():
                # no edge with itself
                if v1 != v2 and v1.aligned_set.share_edge(v2.aligned_set):
                    # add edge between set1 and set2
                    self.connect_vertices(v1, v2)

    def find_graph_vertex_that_has_point(self, x: T.List[float]):
        for vertex in self.vertices.values():
            if vertex.aligned_set.point_is_in_set(x):
                return vertex.name
        assert False, "point is inside no set, " + str(x)

    ################################################################################
    # Graph and plotting utils

    def add_vertex(self, vertex: VertexAlignedSet):
        if vertex.name not in self.vertices:
            self.vertices[vertex.name] = vertex

    def connect_vertices(
        self, left_vertex: Vertex, right_vertex: Vertex, name: str = ""
    ) -> None:
        if name == "":
            name = left_vertex.name + " -> " + right_vertex.name
        edge = Edge(left_vertex, right_vertex, name=name)
        self.add_edge(edge)

    def add_edge(self, edge: Edge) -> None:
        if edge.name not in self.edges:
            self.edges[edge.name] = edge
            edge.right.add_edge_in(edge.name)
            edge.left.add_edge_out(edge.name)

    def plot_the_tessellation_graph(self):
        ax = self.tessellation.plot_the_tessellation(False)
        for edge in self.edges.values():
            ax.add_patch(
                patches.Arrow(
                    edge.left.aligned_set.center[0],
                    edge.left.aligned_set.center[1],
                    edge.right.aligned_set.center[0] - edge.left.aligned_set.center[0],
                    edge.right.aligned_set.center[1] - edge.left.aligned_set.center[1],
                    width=0.1,
                    edgecolor="blue",
                    zorder=120,
                )
            )
        plt.show()


if __name__ == "__main__":
    tess, _, _ = make_a_test_with_objects_and_obstacles()
    # tess = make_swap_two_test()
    graph = GraphOfAdjacentAlignedSets(tess)
    graph.plot_the_tessellation_graph()
    print(graph.vertices.keys())
