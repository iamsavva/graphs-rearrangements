import typing as T

class Vertex:
    """ A simple parent vertex class """
    def __init__(self, name: str)->None:
        # name of the vertex
        self.name = name  # type: str
        # strings that name inbound edges
        self.edges_in = []  # type: T.List[str]
        # string that name outbound edges
        self.edges_out = []  # type: T.List[str]

    def add_edge_in(self, nbh: str)->None:
        assert nbh not in self.edges_in
        self.edges_in.append(nbh)

    def add_edge_out(self, nbh: str)->None:
        assert nbh not in self.edges_out
        self.edges_out.append(nbh)

class Edge:
    """ A simple parent Edge class """
    def __init__(
        self, left_vertex: Vertex, right_vertex: Vertex, name: str
    )->None:
        # TODO: should left / right be strings -- a name?
        self.left = left_vertex # type: Vertex
        self.right = right_vertex # type: Vertex
        self.name = name  # type: str

class Graph:
    """ A simple parent Graph class """
    def __init__(self, vertices: T.Dict[str: Vertex] = {}, edges: T.Dict[str, Edge] = {}):
        self.vertices = vertices # type: T.Dict[str: Vertex]
        self.edges = edges # type: T.Dict[str: Edge]

    def add_vertex(self, vertex: Vertex):
        if vertex.name not in self.vertices:
            self.vertices[vertex.name] = vertex

    def remove_vertex(self, vertex_name: str):
        if vertex_name in self.vertices:
            self.vertices.pop(vertex_name)

    def connect_vertices(self, left_vertex: Vertex, right_vertex: Vertex, name:str = "") -> None:
        if name == "":
            name = left_vertex.name + " -> " + right_vertex.name
        edge = Edge(left_vertex, right_vertex, name=name)
        self.add_edge(edge)

    def add_edge(self, edge: Edge) -> None:
        if edge.name not in self.edges:
            self.edges[edge.name] = edge
            edge.right.add_edge_in(edge.name)
            edge.left.add_edge_out(edge.name)