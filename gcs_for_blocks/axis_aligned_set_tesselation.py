import typing as T

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from axis_aligned_set import Box, AlignedSet
from axis_aligned_set import FREE, OBSTACLE, POSSIBLE_OBJECT, ALIGNED_SET_TYPES
from util import SMALL_TOL

BLOCK_WIDTH = 2.0
HALF_BLOCK_WIDTH = BLOCK_WIDTH / 2.0

class AxisAlignedSetTessellation:
    """
    Given a set of obstacles inside a bounding box, tesselate the space into axis-aligned boxes
    """
    def __init__(self, bounding_box: AlignedSet, obstacles: T.List[AlignedSet], objects: T.List[AlignedSet], object_width: float = BLOCK_WIDTH) -> None:
        self.original_bounding_box = bounding_box.copy()
        self.bounding_box = bounding_box.copy()

        self.width = object_width - SMALL_TOL # aligned tessellation is specified for an object width
        self.half_width = object_width / 2.0 - SMALL_TOL

        # offset the bounding box inwards by (half width - small delta)
        self.bounding_box.offset_inwards(self.half_width)

        self.tessellation_set = set() # type: T.Set[AlignedSet]
        self.tessellation_set.add(self.bounding_box)

        self.tessellate(obstacles + objects)

    def add_to_tessellation(self, new_aligned_set: AlignedSet) -> None:
        """ 
        Add a new axis aligned set to the tessellation. 
        Pre-process the axis aligned set specifically to our object need.
        """
        # pre processing
        new_box = new_aligned_set.copy()
        new_box.offset_inwards(SMALL_TOL)
        new_box.offset_outwards(self.half_width)
        new_box = new_box.intersection(self.bounding_box)

        # finding all sets with which the new box intersects

        new_sets, rem_sets = [], []
        # for each box that's already in the tesselation
        for box_in_tess in self.tessellation_set:
            # no point in trying to intersection an obstacle with anything else -- 
            # it just splits up the obstacle unnecessarily
            if not box_in_tess.is_obstacle():
                # if obstacle intersects with some box
                if new_box.intersects_with(box_in_tess):
                    # remove that box
                    rem_sets.append(box_in_tess)
                    # get 5 direction sets for the new_box
                    direction_sets_for_new_box = new_box.get_direction_sets(self.bounding_box)
                    # add their intersections
                    for dir_set in direction_sets_for_new_box:
                        if box_in_tess.intersects_with(dir_set):
                            intersection_set = box_in_tess.intersection(dir_set)
                            new_sets.append(intersection_set)

        # TODO: in theory if new_box is an obstacle and it was split up into multiple sets, 
            # i can merge them into 1 here.
        # add and remove relevant sets
        for add_me in new_sets:
            self.tessellation_set.add(add_me)
        for rem in rem_sets:
            self.tessellation_set.remove(rem)

    def tessellate(self, objects: T.List[AlignedSet]) -> None:
        """ Generate a dictionary of aligned sets """
        # we will add a new object / obstacle, and only split those sets with which the new obstacle / object intersects
        for new_box in objects:
            self.add_to_tessellation(new_box)

        self.check_tessellation_validity()

    ##############################################################################
    # Supporting functions

    def get_tessellation_dict(self) -> T.Dict[str, AlignedSet]:
        """ Return a dictionary with tessellation names """
        tessellation_list = list(self.tessellation_set)
        tessellation_dict = dict()
        index = 0
        for s in tessellation_list:
            s.name, index = "r" + str(index), index + 1
            tessellation_dict[s.name] = s
        return tessellation_dict

    def check_tessellation_validity(self) -> None:
        tessellation_list = list(self.tessellation_set)
        # check yourself: assert that no sets intersect
        for i, x in enumerate(tessellation_list):
            for j, y in enumerate(tessellation_list):
                if i < j:
                    assert not x.intersects_with(y), "\n" + x.__repr__() + "\n" + y.__repr__()
    
    def plot_the_tessellation(self):
        tessellation_dict = self.get_tessellation_dict()

        _, ax = plt.subplots()

        plt_bounding_box = self.bounding_box.copy()
        plt_bounding_box.offset_outwards(0.1)
        ax.add_patch( plt_bounding_box.get_patch("red", 40) )

        # for each set
        for a_set in tessellation_dict.values():
            # if it's an obstacle -- make it grey, else -- white
            if a_set.is_obstacle():
                color = "grey"
                zorder = 0
            elif a_set.is_object():
                color = "gold"
                zorder = 50
            else:
                color = "white"
                zorder = 100
            ax.add_patch(a_set.get_patch(color, zorder))
            # add a name and a set of obstacles that it represents
            a_set.annotate(ax)

        # axis stuff
        ax.set_xlim([self.original_bounding_box.l, self.original_bounding_box.r])
        ax.set_ylim([self.original_bounding_box.b, self.original_bounding_box.a])
        ax.axis("equal")
        plt.show()
        return ax
        

# TODO: wtf is this function
def get_obstacle_to_set_mapping(
    start_block_pos: T.List[npt.NDArray],
    target_block_pos: T.List[npt.NDArray],
    convex_set_tesselation: T.Dict[str, AlignedSet],
) -> T.Dict[str, str]:
    """
    Block-specific function; per obstacle: find a set in which it is located.
    """
    obstacle_to_set = dict()
    for (i, pos) in enumerate(start_block_pos):
        found = False
        for aset in convex_set_tesselation.values():
            # if point is in the set -- that's it
            if aset.point_is_in_set(pos):
                obstacle_to_set["s" + str(i)] = aset.name
                found=True
                break
        assert found, "no set for location " + str(pos) + " start" + str(i)


    for (i, pos) in enumerate(target_block_pos):
        found = False
        for aset in convex_set_tesselation.values():
            # if point is in the set -- that's it
            if aset.point_is_in_set(pos):
                obstacle_to_set["t" + str(i)] = aset.name
                found=True
                break
        assert found, "no set for location " + str(pos) + " target" + str(i)
    return obstacle_to_set

# helper functions for making aligned sets

def start_loc_aligned_set(x, y, index, half_width = HALF_BLOCK_WIDTH):
    # reduce block width by a delta
    w = half_width
    return AlignedSet(l=x - w, r=x + w, b=y - w, a=y + w, set_type = POSSIBLE_OBJECT, objects = [("s", index)])

def targt_loc_aligned_set(x, y, index, half_width = HALF_BLOCK_WIDTH):
    w = half_width
    return AlignedSet(l=x - w, r=x + w, b=y - w, a=y + w, set_type = POSSIBLE_OBJECT, objects = [("t", index)])

def obstacle_loc_aligned_set(x, y, half_width = HALF_BLOCK_WIDTH):
    w = half_width
    return AlignedSet(l=x - w, r=x + w, b=y - w, a=y + w, set_type = OBSTACLE)

def obstacle_aligned_set(l, r, b, a):
    return AlignedSet(l=l, r=r, b=b, a=a, set_type = OBSTACLE)


def make_swap_two_test():
    # we expect to see three sets stacked on top of each other
    bounding_box = AlignedSet(b=0, a=4, l=0, r=4, set_type=FREE)

    object_sets = []
    object_sets.append( start_loc_aligned_set(2, 1, index = 1))
    object_sets.append( targt_loc_aligned_set(2, 3, index = 1))

    object_sets.append( start_loc_aligned_set(2, 3, index = 2))
    object_sets.append( targt_loc_aligned_set(2, 1, index = 2))

    obstacle_sets = []
    return AxisAlignedSetTessellation(bounding_box, obstacle_sets, object_sets)

def make_swap_two_with_side_test():
    # we expect at least 4 sets
    bounding_box = AlignedSet(b=0, a=4, l=0, r=4, set_type=FREE)

    object_sets = []
    object_sets.append( start_loc_aligned_set(1, 1, index = 1))
    object_sets.append( targt_loc_aligned_set(1, 3, index = 1))

    object_sets.append( start_loc_aligned_set(1, 3, index = 2))
    object_sets.append( targt_loc_aligned_set(1, 1, index = 2))

    obstacle_sets = []
    return AxisAlignedSetTessellation(bounding_box, obstacle_sets, object_sets)

def make_a_test_with_obstacles():
    # we expect at least 4 sets
    bounding_box = AlignedSet(b=0, a=8, l=0, r=8, set_type=FREE)

    object_sets = []
    object_sets.append( start_loc_aligned_set(1, 7, index = 1))
    object_sets.append( targt_loc_aligned_set(7, 1, index = 1))

    object_sets.append( start_loc_aligned_set(1, 1, index = 2))
    object_sets.append( targt_loc_aligned_set(7, 7, index = 2))

    obstacle_sets = []
    obstacle_sets.append( obstacle_aligned_set(0,3,4,6) ) # l r b a
    obstacle_sets.append( obstacle_aligned_set(5,8,2,4) ) # l r b a
    return AxisAlignedSetTessellation(bounding_box, obstacle_sets, object_sets)

if __name__ == "__main__":

    tess = make_swap_two_with_side_test()

    tess.plot_the_tessellation()