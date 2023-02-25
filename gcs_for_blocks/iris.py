# pyright: reportMissingImports=false
import typing as T

import time

from pydrake.geometry.optimization import (  # pylint: disable=import-error
    HPolyhedron,
    Iris,
)
from pydrake.common import RandomGenerator  # pylint: disable=import-error

from .util import WARN, INFO


def sampling_based_IRIS_tesselation(
    obstacles: T.List[HPolyhedron],
    domain: HPolyhedron,
    max_num_sets: int = 9,
    max_num_samples: int = 100,
    verbose=False,
) -> T.List[HPolyhedron]:
    """
    Performing sampling based tesselation of the domain using IRIS.
    Samples points in the domain; if point not in an obstacle or an already existing IRIS set --
    grows an IRIS region out of it.

    Args:
        obstacles (T.List[HPolyhedron]): set of convex obstacles.
        domain (HPolyhedron): domain from in which  are sampled. Must have an interior point.
        max_num_sets (int, optional): return after this many regions have been acquired.
        max_num_samples (int, optional): return after attempting to sample this many regions.
        verbose(bool, optional): verbose the runtime of the tesselation.

    Returns:
        T.List[HPolyhedron]: list of IRIS regions
    """
    sample_counter = 0
    previous_sample = None
    convex_sets = []
    generator = RandomGenerator()
    time_start = time.time()
    while sample_counter < max_num_samples:
        # sample a point
        sample_counter += 1
        if previous_sample is None:
            new_sample = domain.UniformSample(generator)
        else:
            new_sample = domain.UniformSample(
                generator, previous_sample=previous_sample
            )
        previous_sample = new_sample
        # check that a sampled point is not in any obstacle or in already attained set
        sample_not_inside_obstacle_or_existing_sets = True
        for some_set in obstacles + convex_sets:
            if some_set.PointInSet(new_sample):
                sample_not_inside_obstacle_or_existing_sets = False
                break
        if not sample_not_inside_obstacle_or_existing_sets:
            continue
        # find and add an IRIS set
        convex_set = Iris(obstacles, new_sample, domain)
        convex_sets.append(convex_set)
        # check if enough sets have been sampled
        if len(convex_sets) == max_num_sets:
            if verbose:
                INFO("IRIS returned because found max number of IRIS sets.")
            break
    if verbose:
        INFO("IRIS took", time.time() - time_start, "seconds.")
    if sample_counter == max_num_samples and verbose:
        INFO("IRIS returned because sampled max number of points.")
    if len(convex_sets) == 0:
        WARN("IRIS couldn't find a single region!")
    return convex_sets


# def handbuilt_iris_regions(self, num_other_blocks, block_dim):
#     num_blocks = num_other_blocks + 1
#     state_dim = num_blocks * block_dim

#     letter_options = ['a', 'b', 'l', 'r']

# for each letter sequence
# add constraint per block
