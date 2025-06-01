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
import scipy.stats
from tqdm import tqdm
from typing import List


def _iterative_cov(a_u: np.ndarray, n_computation_splits: int = 20) -> np.ndarray | List[float]:
    """
    Compute covariance matrix for d-dimensional normal random sequences. Each component is assumed to be
    Normal(mu=0, sigma=1)

    Parameters
    ----------
    a_u : np.ndarray
        array of uniform numbers
    n_computation_splits : int, optional
        number of segments into which the point set is split for covariance computation

    Returns
    -------
    {np.ndarray, List[float]}
        covariate matrix

    References
    -----------
    [1] Construction and comparison of high-dimensional Sobol’ sequence generators.
        Kucherenko, Sergei & Asotsky, Danil & Atanassov, E. & Roy, Pamphile. (2021).
    """

    m, n = a_u.shape[0], a_u.shape[1]
    a_normal_sequence = scipy.stats.norm.ppf(a_u)
    if n_computation_splits is not None:
        i_splits_lst = list(np.arange(0, n, n / n_computation_splits, dtype=int))
        if i_splits_lst[-1] != n:
            i_splits_lst = i_splits_lst + [n]
        _cov_lst = []
        for start, stop in zip(i_splits_lst[:-1], i_splits_lst[1:]):
            _cov_lst += list(a_normal_sequence.T[start:stop].dot(a_normal_sequence) / m)
    else:
        _cov_lst = np.cov(a_normal_sequence, rowvar=False, ddof=0, dtype=None)
    return _cov_lst


def _compute_sequence_average_correlations(a_u: np.ndarray, max_dim: int = None, n_computation_splits: int = 20,
                                           verbose: bool = False) \
        -> np.ndarray:
    """
    Computation of all the average correlations up to max_dim for d-dimensional normal random sequences. Each component
    is assumed to be Normal(mu=0, sigma=1)

    Parameters
    ----------
    a_u : numpy.ndarray
        array of uniform numbers
    max_dim : int
        maximum dimension for which to compute the average correlation for
    n_computation_splits : int, optional
        number of blocks into which to split the covariance computation
    verbose: bool, optional
       verbosity flag

    Returns
    -------
    numpy.ndarray
        spurious variance value array

    References
    -----------
    [1] Construction and comparison of high-dimensional Sobol’ sequence generators.
        Kucherenko, Sergei & Asotsky, Danil & Atanassov, E. & Roy, Pamphile. (2021).

    """
    if max_dim is None:
        max_dim = a_u.shape[1]

    cov_matrix = _iterative_cov(a_u, n_computation_splits=n_computation_splits)

    a_spurious_variance = np.zeros(max_dim + 1)
    iterable = tqdm(range(1, max_dim + 1)) if verbose else range(1, max_dim + 1)
    for dim in iterable:
        a_spurious_variance[dim] = a_spurious_variance[dim - 1] * (dim - 1)
        a_spurious_variance[dim] += np.array(2 * cov_matrix[dim - 1][: dim - 1]).sum()
        a_spurious_variance[dim] = a_spurious_variance[dim] / dim

    return a_spurious_variance[1:]
