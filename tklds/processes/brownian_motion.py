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
from scipy.stats import norm


def _normal_increments(a_mu: np.ndarray, a_cov: np.ndarray, a_u: np.ndarray) -> np.ndarray:
    """
    Generate multivariate gaussian samples

    Parameters
    ----------
    a_mu: numpy.ndarray
        the mean vector
    a_cov: numpy.ndarray
        the covariance matrix
    a_u: numpy.ndarray
        the uniform random variates

    Returns
    -------
    numpy.ndarray
        increments - multivariate gaussian samples
    """
    a_c = np.linalg.cholesky(a_cov)
    increments = a_mu + (norm.ppf(a_u) @ a_c.T)
    return increments


def _multivariate_brownian_motion(a_x0: np.ndarray, a_mu: np.ndarray, a_cov: np.ndarray, a_u: np.ndarray) -> np.ndarray:
    """
    Simulation a multivariate brownian motion stochastic process

    Parameters
    ----------
    a_x0 : numpy.ndarray
        initial value of path, 2d array of shape (1, num_dims)
    a_mu : numpy.ndarray
        drift, 2d array of shape (1, num_dims)
    a_cov : numpy.ndarray
        covariance matrix, 2d array of shape (num_dims, num_dims)
    a_u : numpy.ndarray
        uniform random variates, 2d array of shape (num_steps, num_dims)

    Returns
    -------
    numpy.ndarray
        sample of paths from the process

    Notes
    -----
    Applying the formula for Ito diffusion:
    dX_t = mu dt + sigma dW_t

    dW_t is a Wiener process, where each sample is Gaussian distributed.
    An inverse CDF transform is used to transform each uniform sample to a normal distribution (dW_t).

    sigma is obtained by applying a Cholesky decomposition to a covariance matrix
    """

    ndim = a_mu.shape[1]
    if a_mu.shape[0] != 1:
        raise ValueError(f"Invalid number of rows of 'mu', expected 1, actual: {a_mu.shape[0]}.")
    if a_cov.shape != (ndim, ndim):
        raise ValueError(f"Invalid 'cov' shape, expected: ({ndim},{ndim}), actual: {a_cov.shape}.")
    if a_u.shape[1] != ndim:
        raise ValueError(f"Invalid number of columns of 'u', expected {ndim}, actual: {a_u.shape[1]}.")
    if np.min(a_u) <= 0 or np.max(a_u) >= 1:
        u_min, u_max = np.min(a_u), np.max(a_u)
        error = f"Invalid uniform variates, expected values in range (0,1) [exclusive], actual: {u_min}, {u_max}."
        raise ValueError(error)
    try:
        normal_increments = _normal_increments(a_mu=a_mu, a_cov=a_cov, a_u=a_u)
        x = a_x0 + np.cumsum(normal_increments, axis=0)
    except Exception as e:
        error = "Error while simulating Brownian motion."
        raise Exception(error) from e
    return x
