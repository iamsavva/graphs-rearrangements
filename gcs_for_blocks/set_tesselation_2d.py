#!/usr/bin/env python3
# pyright: reportMissingImports=false
import typing as T

import numpy as np
import numpy.typing as npt

from pydrake.geometry.optimization import HPolyhedron

from .gcs_options import GCSforAutonomousBlocksOptions
from .util import (
    WARN,
    INFO,
    all_possible_combinations_of_items,
    timeit,
    ChebyshevCenter,
)

from tqdm import tqdm


class SetTesselation:
    def __init__(self, options: GCSforAutonomousBlocksOptions):
        self.opt = options
        self.sets_in_rels_representation = self.get_sets_in_rels_representation()

        self.index2relation = dict()  # T.Dict[int, (int,int)]
        self.make_index_to_block_relation()

        self.rels2set = dict()  # T.Dict[str, HPolyhedron]
        self.generate_sets()

    def get_sets_in_rels_representation(self):
        return all_possible_combinations_of_items(self.opt.rels, self.opt.rels_len)

    def generate_sets(self):
        for i in tqdm(range(len(self.sets_in_rels_representation)), "Set generation"):
            rels_rep = self.sets_in_rels_representation[i]
            # get set
            set_for_rels_rep = self.get_set_for_rels(rels_rep)
            # DO NOT reduce iequalities, some of these sets are empty
            # reducing inequalities is also extremely time consuming
            # set_for_rels_rep = set_for_rels_rep.ReduceInequalities()

            # check that it's non-empty
            solved, x, r = ChebyshevCenter(set_for_rels_rep)
            if solved and r >= 0.00001:
                self.rels2set[rels_rep] = set_for_rels_rep

    def get_set_for_rels(self, rels: str) -> HPolyhedron:
        A, b = self.get_bounding_box_constraint()
        for relation_index, relation in enumerate(rels):
            if relation != "X":
                i, j = self.index2relation[relation_index]
                A_relation, b_relation = self.get_constraints_for_relation(
                    relation, i, j
                )
                A = np.vstack((A, A_relation))
                b = np.hstack((b, b_relation))
        return HPolyhedron(A, b)

    def get_constraints_for_relation(self, relation: str, i: int, j: int):
        if self.opt.symmetric_set_def:
            return self.get_constraints_for_relation_sym(relation, i, j)
        else:
            return self.get_constraints_for_relation_asym(relation, i, j)

    def get_constraints_for_relation_asym(self, relation: str, i, j):
        assert (
            relation != "X"
        ), "Shouldn't be calling get_constraints_for_relation_asym on X"
        w = self.opt.block_width
        bd = self.opt.block_dim
        A = np.zeros((2, self.opt.state_dim))
        if relation == "A":
            A[0, j * bd], A[0, i * bd] = 1, -1
            A[1, j * bd + 1], A[1, i * bd + 1] = 1, -1
            b = np.array([w, -w])
        elif relation == "B":
            A[0, i * bd], A[0, j * bd] = 1, -1
            A[1, i * bd + 1], A[1, j * bd + 1] = 1, -1
            b = np.array([w, -w])
        elif relation == "L":
            A[0, i * bd], A[0, j * bd] = 1, -1
            A[1, j * bd + 1], A[1, i * bd + 1] = 1, -1
            b = np.array([-w, w])
        elif relation == "R":
            A[0, j * bd], A[0, i * bd] = 1, -1
            A[1, i * bd + 1], A[1, j * bd + 1] = 1, -1
            b = np.array([-w, w])
        return A, b

    def get_constraints_for_relation_sym(self, relation: str, i, j):
        assert (
            relation != "X"
        ), "Shouldn't be calling get_constraints_for_relation_sym on X"
        w = self.opt.block_width
        bd = self.opt.block_dim
        sd = self.opt.state_dim
        xi, yi = i * bd, i * bd + 1
        xj, yj = j * bd, j * bd + 1
        a0, a1, a2 = np.zeros(sd), np.zeros(sd), np.zeros(sd)
        if relation == "L":
            a0[xi], a0[yi], a0[xj], a0[yj] = 1, -1, -1, 1
            a1[xi], a1[yi], a1[xj], a1[yj] = 1, 1, -1, -1
            a2[xi], a2[yi], a2[xj], a2[yj] = 1, 0, -1, 0
        elif relation == "A":
            a0[xi], a0[yi], a0[xj], a0[yj] = 1, -1, -1, 1
            a1[xi], a1[yi], a1[xj], a1[yj] = -1, -1, 1, 1
            a2[xi], a2[yi], a2[xj], a2[yj] = 0, -1, 0, 1
        elif relation == "R":
            a0[xi], a0[yi], a0[xj], a0[yj] = -1, 1, 1, -1
            a1[xi], a1[yi], a1[xj], a1[yj] = -1, -1, 1, 1
            a2[xi], a2[yi], a2[xj], a2[yj] = -1, 0, 1, 0
        elif relation == "B":
            a0[xi], a0[yi], a0[xj], a0[yj] = -1, 1, 1, -1
            a1[xi], a1[yi], a1[xj], a1[yj] = 1, 1, -1, -1
            a2[xi], a2[yi], a2[xj], a2[yj] = 0, 1, 0, -1
        A = np.vstack((a0, a1, a2))
        b = np.array([0, 0, -w])
        return A, b

    def get_bounding_box_constraint(self) -> T.Tuple[npt.NDArray, npt.NDArray]:
        A = np.vstack((np.eye(self.opt.state_dim), -np.eye(self.opt.state_dim)))
        b = np.hstack((self.opt.ub, -self.opt.lb))
        return A, b

    def make_index_to_block_relation(self):
        """
        Imagine the matrix of relations: 1-n against 1-n
        There are a total of n-1 relations
        0,1  0,2  0,3 .. 0,n-1
             1,2  1,3 .. 1,n-1
                      ..
                      ..
                         n-2,n-1
        index of the relation is sequential, goes left to right and down.
        """
        st = [0, 1]
        index = 0
        while index < self.opt.rels_len:
            self.index2relation[index] = (st[0], st[1])
            index += 1
            st[1] += 1
            if st[1] == self.opt.num_blocks:
                st[0] += 1
                st[1] = st[0] + 1
        assert st == [self.opt.num_blocks - 1, self.opt.num_blocks], "checking my math"

    def construct_rels_representation_from_point(
        self, point: npt.NDArray, expansion=None
    ) -> str:
        """
        Given a point, find a string of relations for it
        """
        if expansion == None:
            expansion = "Y" * self.opt.rels_len

        rels_representation = ""
        for index in range(self.opt.rels_len):
            # if expansion is an X -- don't do anything
            if expansion[index] == "X":
                rels_representation += "X"
                continue
            i, j = self.index2relation[index]
            for relation in self.opt.rels:
                A, b = self.get_constraints_for_relation(relation, i, j)
                if np.all(A.dot(point) <= b):
                    rels_representation += relation
                    break
        # check yourself -- should have n*(n-1)/2 letters in the representation
        assert len(rels_representation) == self.opt.rels_len
        return rels_representation

    def get_1_step_neighbours(self, rels: str):
        """
        Get all 1 step neighbours
        1-step -- change of a single relation
        """
        assert "X" not in rels, "Un-grounded relation in relation string!"
        assert len(rels) == self.opt.rels_len, "inappropriate relation: " + rels
        lrels = list(rels)
        nbhd = []
        for i in range(len(rels)):
            for j in range(self.opt.number_of_relations - 1):
                lrels[i] = self.opt.rel_iter(lrels[i])
                if (
                    self.opt.rel_inv(rels[i]) != lrels[i]
                    and "".join(lrels) in self.rels2set
                ):
                    nbhd += ["".join(lrels)]
            lrels[i] = self.opt.rel_iter(lrels[i])
        return nbhd

    def get_useful_1_step_neighbours(self, rels: str, target: str):
        """
        Get 1-stop neighbours that are relevant given the target node
        1-step -- change in a single relation
        relevant to target -- if relation in relation is already same as in target, don't change it
        """
        assert "X" not in rels, "Un-grounded relation in relation string! " + rels
        assert len(rels) == self.opt.rels_len, "Wrong num of relations: " + rels
        assert len(target) == self.opt.rels_len, (
            "Wrong num of relations in target: " + target
        )

        nbhd = []
        for i in range(len(rels)):
            if rels[i] == target[i]:
                continue
            elif target[i] in self.opt.rel_nbhd[rels[i]]:
                nbhd += [rels[:i] + target[i] + rels[i + 1 :]]
            else:
                for let in self.opt.rel_nbhd[rels[i]]:
                    nbhd += [rels[:i] + let + rels[i + 1 :]]
        return nbhd
