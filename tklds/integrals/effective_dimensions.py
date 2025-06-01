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
import scipy.integrate
from typing import Callable, List
from tklds.constant import EffectiveDimensionNum


def mean_dimension_additive_superposition(g_lst: List[Callable]) -> float:
    """
    Calculates mean dimension for additive functions in the superposition sense.

    Parameters
    ----------
    g_lst : List[Callable]
        list of test functions for which the mean dimension is to be calculated

    Returns
    -------
    float
        mean dimension in the superposition sense
    """
    return 1.0


def mean_dimension_additive_truncation(g_lst: List[Callable]) -> float:
    """
    Calculates mean dimension for additive functions in the truncation sense.

    Parameters
    ----------
    g_lst : List[Callable]
        list of test functions for which the mean dimension is to be calculated

    Returns
    -------
    mean_dim : float
        mean dimension in the truncation sense
    """

    mu_lst = [scipy.integrate.quad(func=gj, a=0, b=1)[0] for gj in g_lst]
    gamma2_lst = [
        scipy.integrate.quad(func=lambda x: (gj(x) - mu_lst[j]) ** 2, a=0, b=1)[0]
        for j, gj in enumerate(g_lst)
    ]

    mean_dim = np.sum(np.arange(1, len(g_lst) + 1) * gamma2_lst) / np.sum(gamma2_lst)
    return mean_dim


def mean_dimension_multiplicative_superposition(g_lst: List[Callable]) -> float:
    """
    Calculates mean dimension for multiplicative functions in the superposition sense.

    Parameters
    ----------
    g_lst : List[function]
        list of test functions for which the mean dimension is to be calculated

    Returns
    -------
    mean_dim : float
        mean dimension in the superposition sense
    """

    a_mu = np.array([scipy.integrate.quad(func=gj, a=0, b=1)[0] for gj in g_lst])
    gamma2 = np.array(
        [
            scipy.integrate.quad(func=lambda x: (gj(x) - a_mu[j]) ** 2, a=0, b=1)[0]
            for j, gj in enumerate(g_lst)
        ]
    )

    sum_term = np.sum(gamma2 / (gamma2 + a_mu ** 2))
    mult_term = np.prod(a_mu ** 2 / (gamma2 + a_mu ** 2))

    mean_dim = sum_term / (1 - mult_term)
    return mean_dim


def get_dimension_type(mean_dimension: EffectiveDimensionNum) -> Callable:
    """
    Gets the function for calculating a give type of mean effective dimension.

    Parameters
    ----------
    mean_dimension: EffectiveDimensionNum
        type of mean effective dimension to calculate

    Returns
    -------
    f : Callable
        a function for calculating mean effective dimension
    """

    if mean_dimension == EffectiveDimensionNum.ADDITIVE_SUPERPOSITION:
        f = mean_dimension_additive_superposition
    elif mean_dimension == EffectiveDimensionNum.ADDITIVE_TRUNCATION:
        f = mean_dimension_additive_truncation
    elif mean_dimension == EffectiveDimensionNum.MULTIPLICATIVE_SUPERPOSITION:
        f = mean_dimension_multiplicative_superposition
    else:
        raise ValueError("Invalid mean dimension.")
    return f


def mean_effective_dimension_single(g: Callable, mean_dimension: EffectiveDimensionNum,
                                    dimensions: List[int] | np.ndarray) -> List[int]:
    """
    Calculates mean effective dimension when the same function is applied to each dimension.

    Parameters
    ----------
    g : Callable
        test function
    mean_dimension: EffectiveDimensionNum
        type of mean effective dimension to calculate
    dimensions : List[int]
        list of dimensions to iterate

    Returns
    -------
    mu_dim_lst : List[int]
        list of mean effective dimensions
    """
    mean_effective_dimension_fn = get_dimension_type(mean_dimension)
    mu_dim_lst = [mean_effective_dimension_fn([g] * j) for j in dimensions]
    return mu_dim_lst


def mean_effective_dimension_lst(make_g_lst: Callable, mean_dimension: EffectiveDimensionNum,
                                 dimensions_lst: List[int]) -> List[int]:
    """
    Calculates mean effective dimension when a different function is applied to each dimension.

    Parameters
    ----------
    make_g_lst : Callable
        list of functions applied to each dimension
    mean_dimension: EffectiveDimensionNum
        type of mean effective dimension to calculate
    dimensions_lst : List[int]
        list of dimensions to iterate

    Returns
    -------
    mu_dim_lst : List[int]
        list of mean effective dimensions
    """
    mean_effective_dimension_fn = get_dimension_type(mean_dimension)
    mu_dim_lst = [mean_effective_dimension_fn(make_g_lst(j)) for j in dimensions_lst]
    return mu_dim_lst
