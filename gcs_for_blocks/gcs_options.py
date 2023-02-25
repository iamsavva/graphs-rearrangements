import typing as T
import numpy.typing as npt
import numpy as np


class EdgeOptAB:
    """
    Option class for edge connectivity.
    """

    # right point belongs to same set as left point
    add_set_transition_constraint = False
    # right point equal to left point
    add_equality_constraint = False
    # L2 norm on gripper movement
    add_each_block_movement_cost = False

    def __init__(
        self,
        add_set_transition_constraint: bool,
        add_equality_constraint: bool,
        add_each_block_movement_cost: bool,
    ):
        self.add_set_transition_constraint = add_set_transition_constraint
        self.add_equality_constraint = add_equality_constraint
        self.add_each_block_movement_cost = add_each_block_movement_cost

    @staticmethod
    def move_edge() -> "EdgeOptAB":
        return EdgeOptAB(True, False, True)

    @staticmethod
    def equality_edge() -> "EdgeOptAB":
        return EdgeOptAB(False, True, False)

    @staticmethod
    def target_edge() -> "EdgeOptAB":
        return EdgeOptAB(False, False, True)


class EdgeOptions:
    """
    Option class for edge connectivity.
    """

    # right point belongs to orbit of left point in left mode
    add_orbital_constraint = False
    # right point belongs to same set as left point
    add_set_transition_constraint = False
    # right point equal to left point
    add_equality_constraint = False
    # L2 norm on gripper movement
    add_gripper_movement_cost = False
    # add a constant term if performing a mode transition
    add_grasp_cost = False

    def __init__(
        self,
        add_orbital_constraint: bool,
        add_set_transition_constraint: bool,
        add_equality_constraint: bool,
        add_gripper_movement_cost: bool,
        add_grasp_cost: bool,
    ):
        self.add_orbital_constraint = add_orbital_constraint
        self.add_set_transition_constraint = add_set_transition_constraint
        self.add_equality_constraint = add_equality_constraint
        self.add_gripper_movement_cost = add_gripper_movement_cost
        self.add_grasp_cost = add_grasp_cost

    @staticmethod
    def mode_transition_edge(add_grasp_cost: bool) -> "EdgeOptions":
        return EdgeOptions(False, True, True, True, add_grasp_cost)

    @staticmethod
    def within_mode_edge() -> "EdgeOptions":
        return EdgeOptions(True, True, False, True, False)

    @staticmethod
    def between_modes_edge(add_grasp_cost: bool) -> "EdgeOptions":
        return EdgeOptions(True, True, False, True, add_grasp_cost)

    @staticmethod
    def into_in_out_edge() -> "EdgeOptions":
        return EdgeOptions(True, True, False, True, False)

    @staticmethod
    def out_of_in_out_edge() -> "EdgeOptions":
        return EdgeOptions(False, False, True, False, True)

    @staticmethod
    def equality_edge() -> "EdgeOptions":
        return EdgeOptions(False, False, True, False, False)


class GCSforBlocksOptions:
    """
    Option class for GCSforBlocks.
    """

    block_dim: int  # number of dimensions that describe the block world
    num_blocks: int  # number of blocks
    horizon: int  # the GCS is ran in a receding horizon style (akin to a trellis diagram)

    block_width: float = 1.0  # block width

    # connectivity between modes: whether i -> i edges are added.
    allow_self_transitions_for_modes: bool = False
    # allow-self-transitions    -- allow transitioning into itself
    # no-self-transitions       -- don't allow transitioning into itself

    # add a time cost on each edge? this is done to "regularize" the trajectory
    # goal is to reduce possibility of pointlessly grasping and ungrasping in place
    add_grasp_cost: bool = True
    time_cost_weight: float = 1.0  # relative weight between

    # obstacle avoidance
    problem_complexity: str = "obstacles"
    # options:
    # transparent-no-obstacles  -- no collision avoidance, just block movement.
    # transparent-obstacles     -- collision avoidance with pre-defined obstacles but not other
    #                              blocks. not implemented yet
    # obstacles                 -- collision avoidance with both obstacles and other blocks.

    num_gcs_sets: int = -1

    # when solving, this is the max number of rounded paths
    max_rounded_paths: int = 50
    rounding_seed: int = 0
    use_convex_relaxation: bool = True
    custom_rounding_paths: int = 5

    # whether source and target ought to be connected to just one set in the mode
    # TODO: this should really be a default behavior always;
    # it reduces the number of edges and cycles
    connect_source_target_to_single_set: bool = True

    symmetric_set_def: bool = True

    edge_gen: str = "all"

    @property
    def num_modes(self) -> int:
        """
        Number of modes. For the case with no pushing, we have 1 mode for free motion and a mode
        per block for when grasping that block.
        The case with pushing will have many more modes; not implemented.
        """
        return self.num_blocks + 1

    @property
    def state_dim(self) -> int:
        """
        Dimension of the state x optimized at each vertex.
        (number of blocks + gripper) x (dimension of the world)
        """
        return self.num_modes * self.block_dim

    def __init__(
        self,
        block_dim: int,
        num_blocks: int,
        horizon: int,
        block_width: float = 1.0,
        allow_self_transitions_for_modes=False,
        add_grasp_cost: bool = True,
        time_cost_weight: float = 1.0,
        problem_complexity: str = "obstacles",
        max_rounded_paths: int = 40,
        use_convex_relaxation: bool = True,
        connect_source_target_to_single_set: bool = True,
        in_and_out_through_a_single_node: bool = False,
        lb: T.List = None,
        ub: T.List = None,
        lbf: int = None,
        ubf: int = None,
    ):
        assert problem_complexity in ("transparent-no-obstacles", "obstacles")
        self.block_dim = block_dim
        self.num_blocks = num_blocks
        self.horizon = horizon
        self.block_width = block_width
        self.allow_self_transitions_for_modes = allow_self_transitions_for_modes
        self.add_grasp_cost = add_grasp_cost
        self.time_cost_weight = time_cost_weight
        self.problem_complexity = problem_complexity
        self.max_rounded_paths = max_rounded_paths
        self.use_convex_relaxation = use_convex_relaxation
        self.connect_source_target_to_single_set = connect_source_target_to_single_set
        self.in_and_out_through_a_single_node = in_and_out_through_a_single_node
        self.num_gcs_sets = -1

        if lb is not None:
            assert (
                len(lb) == self.block_dim
            ), "Dimension for lower bound constructor must be block_dim"
            self.lb = np.tile(lb, self.num_modes)
        elif lbf is not None:
            self.lb = np.ones(self.state_dim) * lbf
        else:
            self.lb = np.zeros(self.state_dim)

        if ub is not None:
            assert (
                len(ub) == self.block_dim
            ), "Dimension for upper bound constructor must be block_dim"
            self.ub = np.tile(ub, self.num_modes)
        elif ubf is not None:
            self.ub = np.ones(self.state_dim) * ubf
        else:
            self.ub = np.ones(self.state_dim) * 10.0


class GCSforAutonomousBlocksOptions(GCSforBlocksOptions):

    rels = ["A", "B", "L", "R"]
    rel_inverse = {"A": "B", "B": "A", "L": "R", "R": "L"}
    rel_nbhd = {"A": ["L", "R"], "B": ["L", "R"], "L": ["A", "B"], "R": ["A", "B"]}
    rel_iterator = {"A": "B", "B": "L", "L": "R", "R": "A"}

    @property
    def rel_inv(self, letter) -> str:
        return self.rel_inverse[letter]

    @property
    def rel_iter(self, letter) -> str:
        return self.rel_iterator[letter]

    @property
    def number_of_relations(self) -> int:
        return len(self.rels)

    @property
    def rels_len(self) -> int:
        return int((self.num_blocks - 1) * self.num_blocks / 2)

    def paths_from_to(self, start_relation, target_relation):
        if start_relation == target_relation:
            return [[start_relation]]
        elif target_relation in self.rel_nbhd[start_relation]:
            return [[start_relation, target_relation]]
        else:
            paths = []
            for relation in self.rel_nbhd[start_relation]:
                paths += [[start_relation, relation, target_relation]]
            return paths

    @property
    def num_modes(self) -> int:
        """
        Number of modes. For the case with no pushing, we have 1 mode for free motion and a mode
        per block for when grasping that block.
        The case with pushing will have many more modes; not implemented.
        """
        return self.num_blocks

    def __init__(
        self,
        num_blocks: int,
        block_width: float = 1.0,
        max_rounded_paths: int = 40,
        use_convex_relaxation: bool = True,
        lb: T.List = None,
        ub: T.List = None,
        lbf: int = None,
        ubf: int = None,
    ):
        self.block_dim = 2
        self.horizon = -1
        self.num_blocks = num_blocks
        self.block_width = block_width
        self.max_rounded_paths = max_rounded_paths
        self.use_convex_relaxation = use_convex_relaxation
        self.num_gcs_sets = -1

        if lb is not None:
            assert (
                len(lb) == self.block_dim
            ), "Dimension for lower bound constructor must be block_dim"
            self.lb = np.tile(lb, self.num_modes)
        elif lbf is not None:
            self.lb = np.ones(self.state_dim) * lbf
        else:
            self.lb = np.zeros(self.state_dim)

        if ub is not None:
            assert (
                len(ub) == self.block_dim
            ), "Dimension for upper bound constructor must be block_dim"
            self.ub = np.tile(ub, self.num_modes)
        elif ubf is not None:
            self.ub = np.ones(self.state_dim) * ubf
        else:
            self.ub = np.ones(self.state_dim) * 10.0
