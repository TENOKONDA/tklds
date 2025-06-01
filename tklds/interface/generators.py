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

import numpy as np
from tklds.constant import SequenceNum
from tklds.generators.iterative_lds import _load_iterative_lds_tkrg_a_ap5, _load_iterative_lds_joe_kuo, IterativeLDS
from tklds.generators.sobol_engine import SobolEngine


def generate_lds_rvs(sequence: SequenceNum, n: int, d: int, skip: int = 0) -> np.ndarray:
    """
    Gets low discrepancy uniform random variates

    Parameters
    ----------
    sequence: SequenceNum
        sequence to be generated
    n : int
        number of points to get
    d : int
        the number of dimensions of each point
    skip : int, optional
        number of initial points in the sequence to skip

    Returns
    -------
    u : numpy.ndarray
        uniform low discrepancy variates on the interval [0,1)
    """
    if not isinstance(sequence, SequenceNum):
        raise TypeError(f"Expected 'sequence' to be an SequenceNum type, received: {type(SequenceNum).__name__}")
    if not isinstance(n, int):
        raise TypeError(f"Expected 'n' to be an integer type, received: {type(n).__name__}.")
    if not isinstance(d, int):
        raise TypeError(f"Expected 'd' to be an integer type, received: {type(d).__name__}.")
    if n <= 0 or n > 2 ** 32:
        raise ValueError(f"Invalid value {n=}, must be in range [1, 2**32].")
    if d < 1:
        raise ValueError(f"Invalid value {d=}, must be greater than or equal to 1.")

    lds_generator = create_iterative_lds_generator(sequence, d)
    a_u = lds_generator.rvs(size=(n + skip, d))
    return a_u[skip:]


def create_iterative_lds_generator(sequence: SequenceNum, d: int, max_points: int = 2 ** 32) -> IterativeLDS:
    """
    Create an IterativeLDS generator object. This can be used for generating uniform low discrepancy point sequences

    Parameters
    ----------
    sequence : SequenceNum
        sequence to be generated
    d : int
        dimensionality of the generated sequence
    max_points : int, optional
        maximum number of points that the generated can generate. Must be less than or equal to 2 ** 32, after this the
        sequence will wrap around

    Returns
    -------
    iterative_lds : IterativeLDS
    """
    if not isinstance(sequence, SequenceNum):
        raise TypeError(f"Expected 'sequence' to be an SequenceNum type, received: {type(sequence).__name__}")
    if not isinstance(d, int):
        raise TypeError(f"Expected 'd' to be an integer type, received: {type(d).__name__}.")
    if not isinstance(max_points, int):
        raise TypeError(f"Expected 'max_points' to be an integer type, received: {type(max_points).__name__}.")
    if d < 1:
        raise ValueError(f"Invalid value {d=}, must be greater than or equal to 1.")
    if max_points <= 0 or max_points > 2 ** 32:
        raise ValueError(f"Invalid value {max_points=}, must be in range [1, 2**32].")

    if sequence == SequenceNum.TKRG_A_AP5:
        iterative_lds = _load_iterative_lds_tkrg_a_ap5(d=d, max_points=int(max_points))
    elif sequence == SequenceNum.NEW_JOE_KUO:
        iterative_lds = _load_iterative_lds_joe_kuo(d=d, max_points=int(max_points))
    else:
        iterative_lds = _load_iterative_lds_joe_kuo(d=d, max_points=int(max_points))

    return iterative_lds


def create_sobol_lds_engine(sequence: SequenceNum, d: int, scramble: bool = True,
                            seed: int | None | np.random.Generator = None) -> SobolEngine:
    """
    Create a SobolEngine object for generating low discrepancy sequences. The SobolEngine object inherits from
    scipy.qmc.QMCEngine and is compatible with the scipy random number generation and statistics modules.

    Parameters
    ----------
    sequence : SequenceNum
        sequence to be generated
    d: int
        number of dimensions
    scramble : bool, optional
        If True, use Owen scrambling. Otherwise no scrambling is done.
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used, seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is used.

    Returns
    -------
    engine : SobolEngine
    """
    if not isinstance(sequence, SequenceNum):
        raise TypeError(f"Expected 'sequence' to be an SequenceNum type, received: {type(sequence).__name__}")
    if not isinstance(d, int):
        raise TypeError(f"Expected 'd' to be an integer type, received: {type(d).__name__}.")
    if not isinstance(scramble, bool):
        raise TypeError(f"Expected 'scramble' to be a boolean type, received: {type(scramble).__name__}.")
    if d < 1:
        raise ValueError(f"Invalid value {d=}, must be greater than or equal to 1.")

    engine = SobolEngine(d=d, sequence=sequence, scramble=scramble, seed=seed)
    return engine


def get_sequence_enum() -> SequenceNum:
    """
    Get the list of valid low discrepancy direction number sequence names

    Returns
    -------
    SequenceNum
        valid sequences
    """
    return SequenceNum
