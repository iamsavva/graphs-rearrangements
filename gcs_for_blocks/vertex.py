import typing as T

import numpy.typing as npt

from axis_aligned_set import AlignedSet


class Vertex:
    """A simple parent vertex class"""

    def __init__(self, name: str) -> None:
        # name of the vertex
        self.name = name  # type: str
        # strings that name inbound edges
        self.edges_in = []  # type: T.List[str]
        # string that name outbound edges
        self.edges_out = []  # type: T.List[str]

    def add_edge_in(self, nbh: str) -> None:
        assert nbh not in self.edges_in
        self.edges_in.append(nbh)

    def add_edge_out(self, nbh: str) -> None:
        assert nbh not in self.edges_out
        self.edges_out.append(nbh)


class VertexAlignedSet(Vertex):
    """A vertex that also contains a convex set"""

    def __init__(self, name: str, aligned_set: AlignedSet) -> None:
        super().__init__(name)
        # aligned_set
        self.aligned_set = aligned_set

    def get_set_type(self):
        return self.aligned_set.set_type

    def get_objects(self):
        return self.aligned_set.objects

    def get_hpolyhedron_matrices(self) -> T.Tuple[npt.NDArray, npt.NDArray]:
        return self.aligned_set.get_hpolyhedron_matrices()

    def get_perspective_hpolyhedron_matrices(self) -> T.Tuple[npt.NDArray, npt.NDArray]:
        return self.aligned_set.get_perspective_hpolyhedron_matrices()


class VertexTSP(Vertex):
    def __init__(
        self, name: str, block_position: npt.NDArray, possible_object_index: int
    ):
        self.name = name  # type: str
        self.edges_in = []  # type: T.List[str]
        self.edges_out = []  # type: T.List[str]

        self.block_position = block_position  # type: npt.NDArray
        self.possible_object_index = possible_object_index  # type: int


class VertexTSPprogram(VertexTSP):
    def __init__(self, name, block_position, possible_object_index):
        self.name = name  # type: str
        self.edges_in = []  # type: T.List[str]
        self.edges_out = []  # type: T.List[str]
        self.block_position = block_position  # type: npt.NDArray
        self.possible_object_index = possible_object_index

        self.v = (
            None  # visitation variable: whether object-location i is occupied or not
        )
        self.order = None  # tsp-specific: order variable

    @staticmethod
    def from_vertex_tsp(vertex: VertexTSP) -> "VertexTSPprogram":
        return VertexTSPprogram(
            vertex.name, vertex.block_position, vertex.possible_object_index
        )

    def set_v(self, v):
        assert self.v is None, "V for " + self.name + " is already set"
        self.v = v

    def set_order(self, order):
        assert self.order is None, "Order for " + self.name + " is already set"
        self.order = order
