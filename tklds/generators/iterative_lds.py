# A library for Low Discrepancy Sequences developed by the R&D team at TENOKONDA LTD (www.tenokonda.com).
#
# Copyright (c) 2024, TENOKONDA LTD
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder, TENOKONDA LTD, nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os.path
import pickle
from collections import namedtuple
from typing import List

import numpy as np
from tklds.generators.constant import basepath, TKRG_A_AP5_FILEPATH

Dimension = namedtuple('Dimension', ['degree', 'polynomial', 'direction_numbers', 'index'])


def _load_dimensions_list_from_file(filepath: str, drop_implicit: bool = False,
                                    fast_loading: bool = True) -> List[Dimension]:
    """
    Typed loader for direction number files that automatically augments the loaded direction number files with the
    implicit first dimension. Returns a list of dimensions where each dimension of the low discrepancy sequence is
    represented by a `Dimension` object. Dropping of the first, implicit, dimension can be done by asserting
    `drop_implicit`.

    Parameters
    ----------
    filepath: str
        path to direction number file
    drop_implicit: bool, optional
        flag to drop first dimension, implicit direction number file
    fast_loading: bool, optional
        flag to fast loading of dimensions

    Returns
    -------
    dimensions_lst: List[Dimension]
        List of Sobol sequence dimensions. First dimension, implicit in file as automatically added by loader.

    Notes
    -----
    The direction number files start from dimension 2, and the first dimension with values: (0,0,[1]) is implicit.
    This implicit dimension is prepended during loading. If the output of this method is used to construct property
    A/A' matrices - the first element of the returned result must be dropped by asserting `drop_implicit`
    """
    def trim_whitespace(input_str: str) -> str:
        return " ".join(input_str.split())

    if fast_loading:
        with open(filepath + '.pickle', 'rb') as handle:
            dimensions_lst = pickle.load(handle)
    else:
        with open(filepath, "r") as f:
            direction_numbers_lst = f.read()
            direction_numbers_lst = [trim_whitespace(line).split(" ") for line in direction_numbers_lst.split("\n")]
            direction_numbers_lst = direction_numbers_lst[1:-1]  # trim first and last lines

        direction_numbers_lst = [(int(i[1]), int(i[2]), [int(j) for j in i[3:len(i)]]) for i in direction_numbers_lst]
        dimensions_lst = [Dimension(degree=d[0], polynomial=d[1], direction_numbers=d[2], index=i + 2)
                          for i, d in enumerate(direction_numbers_lst)]

    # if we're keeping the first dimension -> add it in!
    if not drop_implicit:
        implicit_dimension = Dimension(degree=32, polynomial=0, direction_numbers=[1] * 32, index=1)
        dimensions_lst = [implicit_dimension] + dimensions_lst

    return dimensions_lst


def _get_first_zero_bit_index(i: int) -> int:
    """
    Gets position of first zero bit in bit string representing integer.

    Parameters
    ----------
    i : int
        input integer

    Returns
    -------
    ci : int
        position of first zero bit
    """
    if i == 0:
        return 1
    else:
        value = i
        ci = 1
        while int(value % 2 == 1) == 1:
            value = np.right_shift(value, 1)
            ci = ci + 1
        return ci


class IterativeLDS:
    """
    Class that performs iterative calculation of low-discrepancy sequences (LDS).

    Attributes
    ----------
    ndim : int
        number of dimensions
    v : numpy.ndarray
        direction number matrix
    current_num_samples : int
        number of samples currently generated
    current_sobol_ints : numpy.ndarray
        Sobol integers calculated so far
    """

    def __init__(self, dimension_lst: List[Dimension], max_points: int = 2 ** 32):
        """
        Initialises IterativeLDS instance.

        Parameters
        ----------
        dimension_lst : List[Dimension]
            list of dimensions
        max_points : int
            maximum number of samples, largest value is 2^32
        """
        self.ndim = len(dimension_lst)

        num_bits = int(np.ceil(np.log2(max_points))) + 1
        self.a_v = np.zeros((self.ndim, num_bits + 1), dtype=np.uintc)
        for i, dim in enumerate(dimension_lst):
            if num_bits <= dim.degree:
                for j in range(1, num_bits + 1):
                    self.a_v[i, j] = np.left_shift(dim.direction_numbers[j - 1], 32 - j)
            else:
                for j in range(1, dim.degree + 1):
                    self.a_v[i, j] = np.left_shift(dim.direction_numbers[j - 1], 32 - j)

                for j in range(dim.degree + 1, num_bits + 1):
                    term = np.right_shift(self.a_v[i, j - dim.degree], dim.degree)
                    self.a_v[i, j] = np.bitwise_xor(self.a_v[i, j - dim.degree], term)

                    for k in range(1, dim.degree):
                        term = np.right_shift(dim.polynomial, (dim.degree - 1 - k)) % 2
                        self.a_v[i, j] = np.bitwise_xor(self.a_v[i, j], term * self.a_v[i, j - k])

        self.current_num_samples = 0
        self.a_current_sobol_ints = np.zeros(self.ndim, dtype=np.uintc)

    def _next_state(self) -> np.ndarray:
        """
        Gets next state, without updating current.

        Returns
        -------
        next_sobol_ints : numpy.ndarray
            next Sobol integers after current ones
        """
        a_next_sobol_ints = np.empty(self.ndim, dtype=np.uintc)
        for i in range(self.ndim):
            a_next_sobol_ints[i] = np.bitwise_xor(
                self.a_current_sobol_ints[i], self.a_v[i, _get_first_zero_bit_index(self.current_num_samples)]
            )
        return a_next_sobol_ints

    def step(self) -> int:
        """
        Advances one step in LDS sequence by updating current state.

        Returns
        -------
        sobol_floats : int
            next Sobol numbers in sequence, where Sobol integers have been shifted right by 32 bits
        """
        sobol_floats = self.a_current_sobol_ints / 2 ** 32
        self.a_current_sobol_ints = self._next_state()
        self.current_num_samples = self.current_num_samples + 1
        return sobol_floats

    def rvs(self, size: int | tuple) -> np.ndarray:
        """
        Generates Sobol sequence iteratively.

        Parameters
        ----------
        size : {int, tuple}
            number of values to be sampled

        Returns
        -------
        u : numpy.ndarray
            samples generated by Sobol sequence
        """
        if isinstance(size, int):
            n = size
        elif isinstance(size, tuple) and len(size) == 1:
            n = size[0]
        elif isinstance(size, tuple) and len(size) == 2:
            if size[1] != self.ndim:
                msg = "Invalid number of dimensions requested"
                raise ValueError(msg)
            n = size[0]
        else:
            msg = f"Invalid size specified: {len(size)}, must either be an int, one-tuple or two-tuple"
            raise ValueError(msg)

        a_u = np.zeros((n, self.ndim))
        for i in range(n):
            a_u[i, :] = self.step()
        return a_u


def _load_iterative_lds_tkrg_a_ap5(d: int, max_points: int = 2 ** 32) -> IterativeLDS:
    """
    Load the 'tkrg-a-ap5' direction numbers into the IterativeLDS generator and return this generator

    Parameters
    ----------
    d : int
        number of dimensions to load
    max_points : int, optional
        maximum number of points that can be generated

    Returns
    -------
    iterative_lds : IterativeLDS
    """
    dimension_lst = _load_dimensions_list_from_file(TKRG_A_AP5_FILEPATH)
    dimension_lst = dimension_lst[:d]

    return IterativeLDS(dimension_lst=dimension_lst, max_points=max_points)


def _load_iterative_lds_joe_kuo(d: int, max_points: int = 2 ** 32) -> IterativeLDS:
    """
    Load the 'new-joe-kuo' direction numbers into the IterativeLDS generator and return this generator

    Parameters
    ----------
    d : int
        number of dimensions to load
    max_points : int, optional
        maximum number of points that can be generated

    Returns
    -------
    iterative_lds : IterativeLDS
    """
    dimension_lst = _load_dimensions_list_from_file(os.path.join(basepath, "direction_numbers", "new-joe-kuo-6.21201"))
    dimension_lst = dimension_lst[:d]

    return IterativeLDS(dimension_lst=dimension_lst, max_points=max_points)
