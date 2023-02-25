# pyright: reportMissingImports=false
import typing as T

import numpy as np
import scipy
import numpy.typing as npt


from pydrake.geometry.optimization import (  # pylint: disable=import-error
    HPolyhedron,
)
from pydrake.solvers import (  # pylint: disable=import-error
    LinearEqualityConstraint,
)

from .util import ERROR, WARN, INFO, YAY
from .iris import sampling_based_IRIS_tesselation
from .gcs_options import GCSforBlocksOptions


class GCSsetGenerator:
    def __init__(self, options: GCSforBlocksOptions):
        self.opt = options

    ###################################################################################
    # Orbital sets and constraints

    def get_orbit_set_for_mode_equality(
        self, mode: int
    ) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """
        When in mode k, the orbit is such that x_m-y_m = 0 for m not k nor 0.
        Produces convex set in a form A [x, y]^T = b
        """
        A = None
        b = None
        d = self.opt.block_dim
        n = self.opt.state_dim
        # for each mode
        for m in range(self.opt.num_modes):
            # that is not 0 or mode
            if m not in (0, mode):
                # add constraint
                A_m = np.zeros((d, 2 * n))
                A_m[:, d * m : d * (m + 1)] = np.eye(d)
                A_m[:, n + d * m : n + d * (m + 1)] = -np.eye(d)
                b_m = np.zeros(d)
                if A is None:
                    A = A_m
                    b = b_m
                else:
                    A = np.vstack((A, A_m))  # type: ignore
                    b = np.hstack((b, b_m))  # type: ignore
        return A, b  # type: ignore

    def get_orbit_set_for_mode_inequality(self, mode: int):
        """
        When in mode k, the orbit is such that x_m-y_m = 0 for m not k nor 0.
        Produces convex set in a form A [x, y]^T <= b
        """
        A, b = self.get_orbit_set_for_mode_equality(mode)
        return self.get_inequality_form_from_equality_form(A, b)

    def get_orbital_constraint(self, mode: int):
        A, b = self.get_orbit_set_for_mode_equality(mode)
        return LinearEqualityConstraint(A, b)

    ###################################################################################
    # Trivial representation transformations

    def get_inequality_form_from_equality_form(
        self, A: npt.NDArray, b: npt.NDArray
    ) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """
        Given a set in a form Ax = b return this same set in a form Ax <= b
        """
        new_A = np.vstack((A, -A))
        new_b = np.hstack((b, -b))
        return new_A, new_b

    ###################################################################################
    # Sets for modes, done clean

    def get_bounding_box_on_x_two_inequalities(
        self,
    ) -> T.Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Bounding box on x is lb <= x <= ub.
        Returns this inequality in a form lb <= Ax <= ub.
        """
        A = np.eye(self.opt.state_dim)
        lb = self.opt.lb
        ub = self.opt.ub
        return A, lb, ub

    def get_bounding_box_on_x_single_inequality(
        self,
    ) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """
        Bounding box on x is lb <= x <= ub.
        Returns this inequality in a form Ax <= b
        """
        A, lb, ub = self.get_bounding_box_on_x_two_inequalities()
        AA = np.vstack((A, -A))
        b = np.hstack((ub, -lb))
        return AA, b

    def get_plane_for_grasping_modes_equality(
        self, mode: int
    ) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """
        When gasping block m, x_0 = x_m. The plane of possible states when in mode k is given by
        x_0 - x_k = 0.
        Returns this plane in the form Ax = b.
        """
        d = self.opt.block_dim
        n = self.opt.state_dim
        A = np.zeros((d, n))
        A[0:d, 0:d] = np.eye(d)
        A[0:d, mode * d : (mode + 1) * d] = -np.eye(d)
        b = np.zeros(d)
        return A, b

    def get_plane_for_grasping_modes_inequality(
        self, mode: int
    ) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """
        When gasping block m, x_0 = x_m. The plane of possible states when in mode k is given by
        x_0 - x_k = 0.
        Returns this plane in the form Ax <= b.
        """
        A, b = self.get_plane_for_grasping_modes_equality(mode)
        return self.get_inequality_form_from_equality_form(A, b)

    def get_convex_set_for_mode_inequality(
        self, mode: int
    ) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """
        Convex set for mode 0 is just the bounding box.
        Convex set for mode k is the bounding box and a plane.
        Returns a convex set for mode in form Ax <= b.
        """
        if mode == 0:
            return self.get_bounding_box_on_x_single_inequality()
        else:
            A_bounding, b_bounding = self.get_bounding_box_on_x_single_inequality()
            A_plane, b_plane = self.get_plane_for_grasping_modes_inequality(mode)
            A = np.vstack((A_bounding, A_plane))
            b = np.hstack((b_bounding, b_plane))
            return A, b

    def get_convex_set_for_mode_polyhedron(self, mode: int) -> HPolyhedron:
        """See get_convex_set_for_mode_inequality"""
        A, b = self.get_convex_set_for_mode_inequality(mode)
        return HPolyhedron(A, b)

    def get_convex_set_experimental(self, free):
        if free == "free":
            A, b = self.get_convex_set_for_mode_inequality(0)
            return HPolyhedron(A, b)
        if free == "grasping":
            # bounding box on state but
            state_dim = self.opt.num_blocks * (self.opt.block_dim + 1)
            A = np.eye(state_dim)
            lb = np.zeros(state_dim)
            ub = np.ones(state_dim)
            ub[0 : self.opt.num_blocks * self.opt.block_dim] *= self.opt.ub[0]

            # single inequality form, bounding box on state
            A = np.vstack((A, -A))
            b = np.hstack((ub, -lb))
            return HPolyhedron(A, b)

    ###################################################################################
    # Obstacles
    def obstacle_in_configuration_space_inequality(
        self, block: int
    ) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """
        When in mode 0, there are no obstacles.
        When in mode m, block m cannot collide with other blocks.
        Other block is given as an obstacle:
            |x_block - x_m| <= block_width
        Since x_m = x_0 in mode k, we have:
            |x_block - x_0| <= block_width

        Returns this obstacle in configuration space as an inequality Ax<=b
        """
        # TODO: should I also add constraint on mode? so both on 0 and on mode?
        # TODO: should I add a boundary?
        d = self.opt.block_dim
        n = self.opt.state_dim
        A = np.zeros((d, n))
        A[:, 0:d] = np.eye(d)
        A[:, block * d : (block + 1) * d] = -np.eye(d)
        b = np.ones(d) * self.opt.block_width
        A = np.vstack((A, -A))
        b = np.hstack((b, b))
        return A, b

    def obstacle_in_configuration_space_polyhedron(self, block: int) -> HPolyhedron:
        """See obstacle_in_configuration_space_inequality"""
        A, b = self.obstacle_in_configuration_space_inequality(block)
        return HPolyhedron(A, b)

    ###################################################################################
    # Mode space transformation

    def transformation_between_configuration_and_mode_space(
        self, mode: int
    ) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """
        Contact-with-block modes are planes in R^n: Ax=b.
        Instead of operating in a n-dimensional space, we can operate on an affine space that is a nullspace of A:
        for x_0 s.t. Ax_0 = b and N = matrix of vectors of the nullspace of A, we have:
        any x, s.t. Ax=b is given by x = x_0 + Ny, where y is of dimension of the nullspace of A.
        This function returns some pair x_0 and N.
        """
        A, b = self.get_plane_for_grasping_modes_equality(mode)
        x_0, residuals = np.linalg.lstsq(A, b, rcond=None)[0:2]
        # print(x_0, residuals)
        # assert np.allclose(residuals, np.zeros(self.opt.state_dim)), "Residuals non zero when solving Ax=b"
        N = scipy.linalg.null_space(A)  # type: ignore
        return x_0, N

    def transformation_between_mode_and_configuration_space(
        self, mode: int
    ) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """
        We can move from mode space into the configuration space using pseudo inverse
        """
        x_0, N = self.transformation_between_configuration_and_mode_space(mode)
        mpi = np.linalg.pinv(N)
        return x_0, mpi

    def configuration_space_inequality_in_mode_space_inequality(
        self, mode: int, A: npt.NDArray, b: npt.NDArray
    ) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """
        Suppose a polyhedron in configuration space is given by Ax <= b
        The mode space for mode is x = x_0 + Ny
        Plugging in, we have obstacle in mode space:
        Ax_0 + ANy <= b
        ANy <= b-Ax_0
        returns AN, b-Ax_0, which define the obstacle in mode space
        """
        # get transformation into mode space
        x_0, N = self.transformation_between_configuration_and_mode_space(mode)
        return A.dot(N), (b - A.dot(x_0))

    def configuration_space_obstacle_in_mode_space(
        self, mode: int, block: int
    ) -> HPolyhedron:
        """
        See inequality_polyhedron_in_mode_space_inequality.
        """
        # get obstacle
        A, b = self.obstacle_in_configuration_space_inequality(block)
        A_m, b_m = self.configuration_space_inequality_in_mode_space_inequality(
            mode, A, b
        )
        return HPolyhedron(A_m, b_m)

    def mode_space_polyhedron_in_configuration_space(
        self, mode: int, poly: HPolyhedron
    ) -> HPolyhedron:
        """
        we can transform polyhedrons in configuration space into polyhedrons in mode space
        """
        A, b = poly.A(), poly.b()
        x_0, mpi = self.transformation_between_mode_and_configuration_space(mode)
        A_c = A.dot(mpi)
        b_c = b + A.dot(mpi.dot(x_0))
        return HPolyhedron(A_c, b_c)

    ###################################################################################
    # Running IRIS

    def get_convex_tesselation_for_mode(self, mode: int) -> T.List[HPolyhedron]:
        """
        NEEDS TESTING
        """

        def combine_sets(
            A_1: npt.NDArray, A_2: npt.NDArray, b_1: npt.NDArray, b_2: npt.NDArray
        ):
            A, b = np.vstack((A_1, A_2)), np.hstack((b_1, b_2))
            return HPolyhedron(A, b)

        # get mode space obstacles
        obstacle_blocks = [i for i in range(1, self.opt.num_modes) if i != mode]
        mode_space_obstacles = [
            self.configuration_space_obstacle_in_mode_space(mode, block)
            for block in obstacle_blocks
        ]
        # get mode space domain
        (
            conf_space_dom_A,
            conf_space_dom_b,
        ) = self.get_bounding_box_on_x_single_inequality()
        # YAY(conf_space_dom_A, conf_space_dom_b)
        (
            mode_space_dom_A,
            mode_space_dom_b,
        ) = self.configuration_space_inequality_in_mode_space_inequality(
            mode, conf_space_dom_A, conf_space_dom_b
        )

        mode_space_domain = HPolyhedron(mode_space_dom_A, mode_space_dom_b)
        mode_space_domain = mode_space_domain.ReduceInequalities()

        # get IRIS tesselation
        mode_space_tesselation = sampling_based_IRIS_tesselation(
            mode_space_obstacles, mode_space_domain
        )
        # move IRIS tesselation into configuration space
        configuration_space_tesselation = [
            self.mode_space_polyhedron_in_configuration_space(mode, poly)
            for poly in mode_space_tesselation
        ]
        # add in-mode constraint to each polyhedron (TODO: this should be redundant?)
        A_mode, b_mode = self.get_convex_set_for_mode_inequality(mode)
        convex_sets_for_mode = [
            combine_sets(A_mode, c.A(), b_mode, c.b())
            for c in configuration_space_tesselation
        ]
        INFO("Iris finished mode", mode)
        return convex_sets_for_mode
