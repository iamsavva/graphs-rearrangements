import typing as T

import numpy as np
import numpy.typing as npt

from graph import Vertex, Edge, Graph
from axis_aligned_set import AlignedSet
from axis_aligned_set_tesselation import AxisAlignedSetTessellation

class VertexAlignedSet(Vertex):
    """ A vertex that also contains a convex set """

    def __init__(self, name: str, aligned_set: AlignedSet)->None:
        super().__init__(name)
        # aligned_set
        self.aligned_set = aligned_set

    def add_edge_in(self, nbh: str)->None:
        assert nbh not in self.edges_in
        self.edges_in.append(nbh)

    def add_edge_out(self, nbh: str)->None:
        assert nbh not in self.edges_out
        self.edges_out.append(nbh)

    def get_set_type(self):
        return self.aligned_set.set_type
    
    def get_objects(self):
        return self.aligned_set.objects

    def get_hpolyhedron_matrices(self) -> T.Tuple[npt.NDArray, npt.NDArray]:
        return self.aligned_set.get_hpolyhedron_matrices()
    
    def get_perspective_hpolyhedron_matrices(self) -> T.Tuple[npt.NDArray, npt.NDArray]:
        return self.aligned_set.get_perspective_hpolyhedron_matrices()

class GraphOfAdjacentAlignedSets(Graph):
    def __init__(self, tess: AxisAlignedSetTessellation):
        super().__init__({},{})
        self.tess = tess

        self.make_graph_from_tessellation()

    def make_graph_from_tessellation(self):
        

