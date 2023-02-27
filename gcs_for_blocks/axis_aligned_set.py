import typing as T

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from util import VERY_SMALL_TOL

FREE = "free"
OBSTACLE = "obstacle"
POSSIBLE_OBJECT = "possible-object"
ALIGNED_SET_TYPES = [FREE, OBSTACLE, POSSIBLE_OBJECT]

class Box:
    """
    Simple class for defining axis aligned boxes and getting their half space representations.
    """

    def __init__(self, lb: npt.NDArray, ub: npt.NDArray, state_dim: int):
        assert state_dim == len(lb)
        assert state_dim == len(ub)
        self.lb = lb
        self.ub = ub
        self.state_dim = state_dim

    def get_hpolyhedron_matrices(self) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """Returns an hpolyhedron for the box"""
        # Ax <= b
        A = np.vstack((np.eye(self.state_dim), -np.eye(self.state_dim)))
        b = np.hstack((self.ub, -self.lb))
        return A, b

    def get_perspective_hpolyhedron_matrices(self) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """Returns a perspective hpolyhedron for the box"""
        # Ax <= b * lambda
        # Ax - b * lambda <= 0
        # [A -b] [x, lambda]^T <= 0
        A, b = self.get_hpolyhedron_matrices()
        b.resize((2 * self.state_dim, 1))
        pA = np.hstack((A, (-1) * b))
        pb = np.zeros(2 * self.state_dim)
        return pA, pb

class AlignedSet:
    """
    A class that defines a 2D axis aligned set and relevant tools.
    """

    def __init__(
        self,
        a: float,
        b: float,
        l: float,
        r: float,
        set_type: str,
        name: str = "",
        objects: T.List[int] = [],
    ) -> None:
        # above bound a, y <= a
        # below bound b, b <= y
        # left  bound l, l <= x
        # right bound r, x <= r
        self.constraints = {"a": a, "b": b, "l": l, "r": r}  # type: T.Dict[str, float]
        # TODO make more descriptive
        self.name = name  # type: str 
        self.box = Box(lb=np.array([l, b]), ub=np.array([r, a]), state_dim=2)  # type: Box

        assert set_type in ALIGNED_SET_TYPES, "bad set type: " + set_type
        self.set_type  = set_type

        if set_type != POSSIBLE_OBJECT:
            assert len(objects) == 0
        else:
            assert len(objects) > 0
        self.objects = objects # objects associated with the 

    def copy(self) -> "AlignedSet":
        return AlignedSet(
            a=self.a, b=self.b, l=self.l, r=self.r, set_type = self.set_type, name=self.name, objects=self.objects
        )
    
    def __repr__(self):
        return self.set_type + ": " "L:" + str(round(self.l, 3)) + " R:" + str(round(self.r, 3)) + " B:" + str(round(self.b, 3)) + " A:" + str(round(self.a, 3))

    ######################################################################
    # Properties and getters

    @property
    def l(self) -> float:
        return self.constraints["l"]

    @property
    def r(self) -> float:
        return self.constraints["r"]

    @property
    def a(self) -> float:
        return self.constraints["a"]

    @property
    def b(self) -> float:
        return self.constraints["b"]
    
    @property
    def center(self) -> npt.NDArray:
        return np.array([ (self.l+self.r)/2.0, (self.a+self.b)/2.0 ])

    def is_object(self):
        return self.set_type == POSSIBLE_OBJECT
    
    def is_obstacle(self):
        return self.set_type == OBSTACLE
    
    def is_free(self):
        return self.set_type == FREE
    
    def get_patch(self, color: str, zorder = 100):
        """ Get a patch for drawing """
        return patches.Rectangle(
            (self.l, self.b),
            self.r - self.l,
            self.a - self.b,
            linewidth=2,
            edgecolor="black",
            facecolor=color,
            zorder = zorder,
            # label=self.name, # TODO: put list of objects instead
        )
    
    def annotate(self, ax: plt.Axes) -> None:
        """ Produce an annotation on top of the axis """
        ax.annotate(
            self.name + "\n" + str(self.objects),
            ((self.l + self.r) / 2, (self.b + self.a) / 2),
            color="black",
            weight="bold",
            fontsize=8,
            ha="center",
            va="center",
            zorder = 200,
        )

    def get_hpolyhedron_matrices(self) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """ Return A and b matrices, Ax <= b """
        return self.box.get_hpolyhedron_matrices()

    def get_perspective_hpolyhedron_matrices(self)-> T.Tuple[npt.NDArray, npt.NDArray]:
        """ Return A and b matrices of the perspective poly, [A, -b] [x, phi] <= 0"""
        return self.box.get_perspective_hpolyhedron_matrices()
    
    def resolve_type(self, t1, t2):
        """
        relevant when we intersect two objects. 
        obstacle intersect anything is obstacle.
        object intersect free or object is object.
        """
        if OBSTACLE in (t1,t2):
            return OBSTACLE
        if POSSIBLE_OBJECT in (t1, t2):
            return POSSIBLE_OBJECT
        return FREE

    
    ######################################################################
    # offsets and intersections

    def offset_inwards(self, delta: float):
        """
        Offset the box inwards (hence making it smaller) -- to make interior non-empty.

        Args:
            delta (float): offset amount
        """
        self.constraints["l"] = self.l + delta
        self.constraints["r"] = self.r - delta
        self.constraints["b"] = self.b + delta
        self.constraints["a"] = self.a - delta

    def offset_outwards(self, delta: float):
        self.offset_inwards(-delta)

    def point_is_in_set(self, point: T.List[float]):
        """True if point is inside a set"""
        return self.l <= point[0] <= self.r and self.b <= point[1] <= self.a

    def intersects_with(self, other: "AlignedSet") -> bool:
        """
        Instead of working with tight bounds, offset all boxes inwards by a small amount.
        Interseciton = fully right of or left of or above or below
        """
        # intersection by edge is fine
        return not (
            self.r <= other.l or other.r <= self.l or self.a <= other.b or other.a <= self.b
        )

    def share_edge(self, other: "AlignedSet", rtol=VERY_SMALL_TOL) -> bool:
        """
        Two sets share an edge if they intersect
            + left of one is right of another  or  below of one is above of another.
        """
        b, a = max(self.b, other.b), min(self.a, other.a)
        l, r = max(self.l, other.l), min(self.r, other.r)
        return ((a - b) > 0 and np.isclose(l, r, rtol)) or ((r - l) > 0 and np.isclose(b, a, rtol))

    def intersection(self, other: "AlignedSet"):
        """Intersection of two sets; cannot be just an edge (i.e., without interior)"""
        assert self.intersects_with(other), "sets don't intersect:\n" + self.__repr__() + "\n" + other.__repr__()
        b, a = max(self.b, other.b), min(self.a, other.a)
        l, r = max(self.l, other.l), min(self.r, other.r)
        set_type = self.resolve_type(self.set_type, other.set_type)
        objects = self.objects + other.objects
        if set_type == OBSTACLE:
            objects = []
            # assert False, "Attempting to intersect an obstacle and an object! Names:\n" + self.__repr__() + "\n" + other.__repr__()

        return AlignedSet(a=a, b=b, l=l, r=r, set_type=set_type, objects=objects)

    def is_inside(self, box: "AlignedSet"):
        return self.l >= box.l and self.r <= box.r and self.b >= box.b and self.a <= box.a

    ######################################################################
    # get a tessellation from the aligned set

    def get_direction_sets(self, bounding_box: "AlignedSet") -> T.List["AlignedSet"]:
        """A box tesselates a space into 5 sets: above / below / left / right / itself"""
        assert self.is_inside(bounding_box)
        dir_sets = []
        # left
        dir_sets.append(AlignedSet(l=bounding_box.l, r=self.l, a=self.a, b=self.b, set_type=FREE))
        # right
        dir_sets.append(AlignedSet(r=bounding_box.r, l=self.r, a=self.a, b=self.b, set_type=FREE))
        # below
        dir_sets.append(AlignedSet(r=bounding_box.r, l=bounding_box.l, a=self.b, b=bounding_box.b, set_type=FREE))
        # above
        dir_sets.append(AlignedSet(r=bounding_box.r, l=bounding_box.l, b=self.a, a=bounding_box.a, set_type=FREE))
        # itself
        dir_sets.append(self.copy())
        return dir_sets