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

from tklds.processes.brownian_motion import _multivariate_brownian_motion
from tklds.processes.spurious_variance import _compute_sequence_average_correlations
from tklds.utilities import _assert_array_is_2d


def spurious_variance(a_u: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Compute the spurious variance from the uniformly distributed inputs

    Parameters
    ----------
    a_u : numpy.ndarray
        the uniform variates for which to compute the spurious variance parameter
    verbose : bool, optional
        produce verbose output

    Returns
    -------
    a_spv : numpy.ndarray
        spurious variance value array

    """
    if not isinstance(a_u, np.ndarray):
        raise TypeError(f"Invalid type 'u'. Expected numpy.ndarray, actual: {type(a_u).__name__}")
    a_spv = _compute_sequence_average_correlations(a_u, max_dim=None, n_computation_splits=20, verbose=verbose)
    return a_spv


def multivariate_brownian_motion(a_x0: np.ndarray, a_mu: np.ndarray, a_cov: np.ndarray, a_u: np.ndarray) -> np.ndarray:
    """
    Generate a multivariate brownian motion stochastic process trajectory

    Parameters
    ----------
    a_x0 : np.ndarray
        initial value
    a_mu : np.ndarray
        normal increments noise mean
    a_cov : np.ndarray
        normal increments covariance matrix
    a_u : np.ndarray
        uniform random variates used for the simulation

    Returns
    -------
    numpy.ndarray
        sample of paths from the process
    """
    if not isinstance(a_x0, np.ndarray):
        raise TypeError(f"Invalid type 'x0'.  Expected numpy.ndarray, actual: {type(a_x0).__name__}")

    if not isinstance(a_cov, np.ndarray):
        raise TypeError(f"Invalid type 'cov'.  Expected numpy.ndarray, actual: {type(a_cov).__name__}")

    if not isinstance(a_mu, np.ndarray):
        raise TypeError(f"Invalid type 'mu'.  Expected numpy.ndarray, actual: {type(a_mu).__name__}")

    if not isinstance(a_u, np.ndarray):
        raise TypeError(f"Invalid type 'u'. Expected numpy.ndarray, actual: {type(a_u).__name__}")

    if not _assert_array_is_2d(a_cov) and a_cov.shape[0] != a_cov.shape[1]:
        raise ValueError(f"Invalid 'cov' input. Expected square covariance matrix: "
                         f"{a_cov.shape[0]} != {a_cov.shape[1]}")

    if a_cov.shape[0] != a_x0.shape[1] or a_x0.shape[1] != a_mu.shape[1] or a_mu.shape[1] != a_u.shape[1]:
        raise ValueError(f"Dimension mismatch for parameters 'cov', 'x0', 'mu', 'u'.")

    if (a_cov != a_cov.T).any():
        raise ValueError(f"Invalid cov input. Expected symmetric covariance matrix.")

    if np.any(np.linalg.eigvals(a_cov) <= 0):
        raise ValueError("Invalid cov input.  Expected covariance matrix to be positive definite.")

    return _multivariate_brownian_motion(a_x0=a_x0, a_mu=a_mu, a_cov=a_cov, a_u=a_u)
