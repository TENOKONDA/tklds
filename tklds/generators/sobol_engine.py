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

from typing import List

import numpy as np
from tklds.constant import SequenceNum
from scipy.stats.qmc import QMCEngine, Sobol
from tklds.generators.constant import TKRG_A_AP5_FILEPATH, JOE_KUO_FILEPATH
from tklds.generators.iterative_lds import _load_dimensions_list_from_file, Dimension


class SobolEngine(Sobol):

    def __init__(self, d: int, sequence: SequenceNum = SequenceNum.TKRG_A_AP5, scramble: bool = True,
                 seed: int | None | np.random.Generator = None):
        """
        Constructor for the SobolEngine object that allows integration of tklds with the scipy random number generation
        framework, allowing the low discrepancy sequences to be used inside scipy.

        Parameters
        ----------
        d : int
            dimension of the generated low discrepancy sequence values
        sequence: SequenceNum, optional
            sequence to be generated
        scramble : bool, optional
            If True, use scrambling. Otherwise no scrambling is done.
        seed : {None, int, `numpy.random.Generator`}, optional
            If `seed` is None the `numpy.random.Generator` singleton is used.
            If `seed` is an int, a new ``Generator`` instance is used, seeded with `seed`.
            If `seed` is already a ``Generator`` instance then that instance is used.
        """

        QMCEngine.__init__(self, d=d, seed=seed)

        self._sequence = sequence
        dimensions_lst = self._load_direction_numbers()

        self.bits = 30
        self.dtype_i = int if hasattr(Sobol, 'MAXBIT') else np.uint32
        self.maxn = 2**self.bits

        self._sv: np.ndarray = np.zeros((self.d, self.bits), dtype=self.dtype_i)
        self._initialize_v(dimensions_lst)

        if not scramble:
            self._shift = np.zeros(d, dtype=self.dtype_i)
        else:
            self._scramble()

        self._quasi = self._shift.copy()
        self._scale = 1.0 / 2 ** self.bits
        self._first_point = (self._quasi / 2 ** self.bits).reshape(1, -1)

    def _load_direction_numbers(self) -> List[Dimension]:
        """
        Loader for direction number. Returns a list of dimensions where each dimension of the low discrepancy
        sequence is represented by a `Dimension` object.

        Returns
        -------
        dimensions_lst: List[Dimension]
            List of Sobol sequence dimensions.
        """
        if self._sequence == SequenceNum.TKRG_A_AP5:
            path = TKRG_A_AP5_FILEPATH
        elif self._sequence == SequenceNum.NEW_JOE_KUO:
            path = JOE_KUO_FILEPATH

        return _load_dimensions_list_from_file(path)[:self.d]

    def _initialize_v(self, dimensions_lst: List[Dimension]):
        """
        Initialize matrix v with direction numbers

        Parameters
        ----------
        dimensions_lst: List[Dimension]
            List of Sobol sequence dimensions.
        """

        if self.d == 0:
            return

        for i in range(self.bits):
            self._sv[0, i] = 1

        for di in range(1, self.d):
            p = 2 * dimensions_lst[di].polynomial + 2 ** dimensions_lst[di].degree + 1
            m = p.bit_length() - 1

            self._sv[di, :m] = dimensions_lst[di].direction_numbers
            for j in range(m, self.bits):
                newv = self._sv[di, j - m]
                pow2 = 1
                for k in range(m):
                    pow2 = pow2 << 1
                    if (p >> (m - 1 - k)) & 1:
                        newv = newv ^ (pow2 * self._sv[di, j - k - 1])
                self._sv[di, j] = newv

        pow2 = 1
        for j in range(self.bits):
            for i in range(self.d):
                self._sv[i, self.bits - 1 - j] *= pow2
            pow2 = pow2 << 1

    def reset(self) -> Sobol:
        """
        Reset the engine to base state.

        Returns
        -------
        engine : Sobol
            Engine reset to its base state.
        """
        super().reset()
        return self

    def fast_forward(self, n: int) -> Sobol:
        """
        Fast-forward the sequence by `n` positions.

        Parameters
        ----------
        n : int
            Number of points to skip in the sequence.

        Returns
        -------
        engine : Sobol
            The fast-forwarded engine.
        """
        return super().fast_forward(n=n)

    def random_base2(self, m: int) -> np.ndarray:
        """
        Draw point(s) from the Sobol' sequence.

        This function draws :math:`n=2^m` points in the parameter space
        ensuring the balance properties of the sequence.

        Parameters
        ----------
        m : int
            Logarithm in base 2 of the number of samples; i.e., n = 2^m.

        Returns
        -------
        sample : array_like (n, d)
            Sobol' sample.
        """
        return super().random_base2(m=m)

    def random(self, n: int = 1) -> np.ndarray:
        """
        Generate low discrepancy sequence points

        Parameters
        ----------
        n : int, optional
            number of points to generated

        Returns
        -------
        sample : array_like (n, d)
            Sobol' sample.
        """
        return super().random(n=n)
