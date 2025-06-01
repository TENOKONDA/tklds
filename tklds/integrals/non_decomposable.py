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

from typing import Tuple, List

import numpy as np
import pandas as pd
import sympy as sp
from scipy import stats as st

from sympy import Piecewise

from tklds.constant import SequenceNum
from tklds.generators.constant import TKRG_A_AP5_FILEPATH, JOE_KUO_FILEPATH
from tklds.generators.iterative_lds import _load_dimensions_list_from_file, IterativeLDS


def genzs_discontinuous_analytic_integral(mu: float, a: float, d: int) -> float:
    """
    Performs analytic integral of Genz's discontinuous function.

    Parameters
    ----------
    mu : float
       mean of the integrand
    a : float
       number that controls importance of each dimension
    d : int
        number of dimensions

    Returns
    -------
    integral : float
        analytic test integral
    """
    ax_lst = []
    x_lst = []
    for i in range(d):
        ax_lst.append(a * sp.symbols(f'x{i}'))
        x_lst.append(sp.symbols(f'x{i}'))

    exponent_analytic = -np.sum(ax_lst)

    # indicator function
    ind_int = [x - mu for x in x_lst]

    res = Piecewise((0, (ind_int[0] <= 0) | (ind_int[1] <= 0)), (sp.exp(exponent_analytic), True))

    ranges_lst = [(x, 0, 1) for x in x_lst]

    integral = sp.integrate(res, *ranges_lst)
    return integral


def genzs_discontinuous_function(a_u: np.ndarray, mu: float, a: float, d: int) -> float:
    """
    Performs analytic integral of Genz's discontinuous function.

    Parameters
    ----------
    a_u : numpy.ndarray
        random samples
    mu  : float
        mean of the integrand
    a : float
        number that controls importance of each dimension
    d : int
        number of dimensions

    Returns
    -------
    integral : float
        analytic test integral
    """
    a_mu = np.array([mu] * d)
    a_a = np.array([a] * d)
    a_exponent_numeric = -np.sum(a_a * a_u, axis=1)

    # indicator function
    a_ind_int = a_u - a_mu

    actual_lst = []
    for i, a_sample in enumerate(a_ind_int):
        if a_sample[0] <= 0 or a_sample[1] <= 0:
            actual_lst.append(0)
        else:
            actual_lst.append(np.exp(a_exponent_numeric[i]))
    actual = np.mean(actual_lst)
    expected = genzs_discontinuous_analytic_integral(mu=mu, a=a, d=d)
    return expected, actual


def atanassov_row_func(a_u: np.ndarray, d: int, is_analytic: bool = True) -> float:
    """
    Defines Atannasov function for a given row.

    Parameters
    ----------
    a_u : numpy.ndarray
        random samples
    d : int
        number of dimensions
    is_analytic : bool
        flag indicating whether integral is analytic

    Returns
    -------
    float
        integral for a given row
    """
    if is_analytic:
        d = a_u.shape[0]
        coeff = (1 + 1 / d)
        return coeff * np.prod(a_u) ** (1 / d)
    coeff = (1 + 1 / d)
    return coeff * np.prod(a_u, axis=1) ** (1 / d)


def atanassov_analytic_integral(d: int) -> float:
    """
    Integrates Atanassov function analytically.

    Parameters
    ----------
    d : int
        number of dimensions

    Returns
    -------
    float
        analytic integral
    """
    x_lst = []
    for i in range(d):
        x_lst.append(sp.symbols(f'x{i}'))

    res = atanassov_row_func(np.array(x_lst), d, is_analytic=True)
    ranges_lst = [(x, 0, 1) for x in x_lst]

    integral = sp.integrate(res, *ranges_lst)
    return integral


def atanassov_function(a_u: np.ndarray, d: int) -> Tuple[float, float]:
    """
    Provides analytic and expected Atanassov integral.

    Parameters
    ----------
    a_u : numpy.ndarray
        random samples
    d : int
        number of dimensions

    Returns
    -------
    expected : float
        analytic Atanassov integral value
    actual : float
        estimated Atanassov integral value
    """
    actual = atanassov_row_func(a_u, d, is_analytic=False)
    actual = np.mean(actual)
    expected = atanassov_analytic_integral(d)
    return expected, actual


def _generate_points(sequence: SequenceNum, num_points: int, num_dim: int, num_skip_points: int = 0) -> np.ndarray:
    """
    Generate the uniform variates using the different low discrepancy sequence values

    Parameters
    ----------
    sequence : SequenceNum
        sequence to generate
    num_points : int
        number of points to generate
    num_dim : int
        number of dimensions to generate
    num_skip_points : int, optional
        number of points to skip

    Returns
    -------
    numpy.ndarray
        the uniform variates returned by the generating process
    """
    if not isinstance(sequence, SequenceNum):
        raise TypeError(f"Expected 'sequence' to be a SequenceNum type, received: {type(sequence).__name__}")

    if sequence == SequenceNum.TKRG_A_AP5:
        dimensions_lst = _load_dimensions_list_from_file(TKRG_A_AP5_FILEPATH)[:num_dim]
        generator = IterativeLDS(dimension_lst=dimensions_lst)
    elif sequence == SequenceNum.NEW_JOE_KUO:
        dimensions_lst = _load_dimensions_list_from_file(JOE_KUO_FILEPATH)[:num_dim]
        generator = IterativeLDS(dimension_lst=dimensions_lst)
    elif sequence == SequenceNum.PCG64:
        generator = st.uniform(0, 1)

    return generator.rvs(size=(int(num_points), num_dim))[num_skip_points:, :]


def _improper_integral_benchmark(num_points_lst: List[int], d: int, verbose: bool = False) -> pd.DataFrame:
    """
    Performs improper integral test on each generator.

    Parameters
    ----------
    num_points_lst : List[int]
        list of sample sizes to be used
    d : int
        number of dimensions
    verbose : bool
        flag indicating whether to print progress bar

    Returns
    -------
    results : pandas.DataFrame
        results dataframe with the following columns:
            "generator", "num_samples", "c"
    """
    df_results = pd.DataFrame(columns=["generator", "num_samples", "c"])

    for sequence in [SequenceNum.TKRG_A_AP5, SequenceNum.NEW_JOE_KUO, SequenceNum.PCG64]:

        c_lst = []
        c = np.inf
        c_tmp = np.inf
        for i, n in enumerate(num_points_lst):
            if verbose:
                print(f"{sequence=}, {n=}, [progress: {i}/{len(num_points_lst)}]", end='\r')
            a_u = _generate_points(sequence=sequence, num_points=n, num_dim=d, num_skip_points=1)
            for a_un in a_u:
                c_tmp = 1
                for u_nd in a_un:
                    c_tmp *= u_nd
            c = min(c, c_tmp)
            c_lst.append(c)
        sub_results_dict = {"num_samples": num_points_lst, "c": c_lst}
        sub_results = pd.DataFrame(sub_results_dict)
        sub_results["generator"] = sequence.value
        df_results = pd.concat([df_results, sub_results], ignore_index=True)
    return df_results
