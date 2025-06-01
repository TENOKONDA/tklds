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
import scipy.special as sp

from tklds.utilities import _assert_array_is_2d


def hellekalek_function(x: float | np.ndarray, alpha: int = 1) -> float | np.ndarray:
    """
    Evaluates the Hellekalek integral.

    Parameters
    ----------
    x : {float, np.ndarray}
        sample on which to evaluate
    alpha : int
        hyperparameter controlling denominator

    Returns
    -------
    {float, np.ndarray}
        result of evaluation
    """
    return (x ** alpha - 1) / (1 + alpha)


def hellekalek_function_comparison(a_u: np.ndarray, alpha: int = 1) -> Tuple[float, float]:
    """
    Calculates the Hellekalek integral.

    Parameters
    ----------
    a_u : numpy.ndarray
        random samples
    alpha : int
        hyperparameter controlling denominator

    Returns
    -------
    actual : float
        test integral based on random samples
    expected : float
        analytic test integral
    """
    _assert_array_is_2d(a_u)
    if not ((alpha >= 1) and (alpha <= 3)):
        msg = f"input alpha values outside range [1,3] with value: {alpha}"
        raise ValueError(msg)

    a_actual = hellekalek_function(x=a_u, alpha=alpha)

    actual = np.mean(np.prod(a_actual, axis=1))

    ndim = a_u.shape[1]
    expected = (-alpha / (1 + alpha) ** 2) ** ndim
    return actual, expected


def sobol_1(x: float | np.ndarray, aj: List[float] | List[int] | int = 1) -> float | np.ndarray:
    """
    Evaluates Sobol function 1 integral.

    Parameters
    ----------
    x : {float, np.ndarray}
        sample on which to evaluate
    aj : {List[float], List[int], int}
        list of numbers, where each aj >= 0, that controls importance of each dimension

    Returns
    -------
    {float, np.ndarray}
        result of evaluation
    """
    return (np.abs((4 * x - 2)) + aj) / (aj + 1)


def sobol_1_comparison(a_u: np.ndarray, aj_lst: List[float] | List[int]) -> Tuple[float, float]:
    """
    Calculates Sobol function 1 integral.

    Parameters
    ----------
    a_u : numpy.ndarray
        random samples
    aj_lst : {List[float], List[int]}
        list of numbers, where each aj >= 0, that controls importance of each dimension

    Returns
    -------
    actual : float
        test integral based on random samples
    expected : float
        analytic test integral
    """
    _assert_array_is_2d(a_u)
    if len(aj_lst) != a_u.shape[1]:
        msg = (
            f"Invalid input, expected number of a_j(={len(aj_lst)}) values"
            f" to equal number of dimensions {a_u.shape[1]}."
        )
        raise ValueError(msg)
    invalid_aj_lst = [aj for aj in aj_lst if aj < 0]
    if len(invalid_aj_lst) > 0:
        msg = f"input contains invalid aj values: {invalid_aj_lst}"
        raise ValueError(msg)

    aj = np.array(aj_lst).reshape(1, len(aj_lst))
    x = sobol_1(x=a_u, aj=aj)
    actual = np.mean(np.prod(x, axis=1))

    return actual, 1


def sobol_2(x: float | np.ndarray, d: int = 1) -> float | np.ndarray:
    """
    Evaluates Sobol function 2 test integral of random samples.

    Parameters
    ----------
    x : {float, np.ndarray}
        sample on which to evaluate
    d : int
        dimensionality for which to evaluate sample

    Returns
    -------
    {float, np.ndarray}
        result of evaluation
    """
    return (d + 2 * x) / (1 + d)


def sobol_2_comparison(a_u: np.ndarray, dimensions: None | int | np.ndarray = None) -> Tuple[float, float]:
    """
    Calculates Sobol function 2 test integral of random samples.

    Parameters
    ----------
    a_u : numpy.ndarray
         random samples
    dimensions : {None, int, numpy.ndarray}, optional
         test dimensions

    Returns
    -------
    actual : float
        test integral based on random samples
    expected : float
        analytic test integral
    """
    _assert_array_is_2d(a_u)
    if dimensions is None:
        dimensions = np.arange(1, a_u.shape[1] + 1).reshape((1, a_u.shape[1]))
    a_actual = sobol_2(x=a_u, d=dimensions)
    actual = np.mean(np.prod(a_actual, axis=1))
    expected = 1
    return actual, expected


def owens_example(x: float | np.ndarray, d: int) -> float | np.ndarray:
    """
    # Evaluates Owens example test integral of random samples.

    Parameters
    ----------
    x : {float, np.ndarray}
        sample on which to evaluate
    d : int
         test dimensions

    Returns
    -------
    {float, np.ndarray}
        result of evaluation
    """
    return 12 ** (1 / 2.0) * (x ** d - 0.5)


def owens_example_comparison(a_u: np.ndarray, dimensions: None | int | np.ndarray = None) -> Tuple[float, float]:
    """
    Calculates Owens example test integral of random samples.

    Parameters
    ----------
    a_u : numpy.ndarray
        random samples
    dimensions : {None, int, numpy.ndarray}, optional
         test dimensions

    Returns
    -------
    actual : float
        test integral based on random samples
    expected : float
        analytic test integral
    """
    if dimensions is None:
        dimensions = np.arange(1, a_u.shape[1] + 1)
    a_actual = owens_example(x=a_u, d=dimensions)

    actual = np.mean(np.prod(a_actual, axis=1))
    expected = 0
    return actual, expected


def roos_and_arnold_2(x: float | np.ndarray) -> float | np.ndarray:
    """
    # Evaluates Roos and Arnold function 2 test integral of random samples.

    Parameters
    ----------
    x : {float, np.ndarray}
        sample on which to evaluate

    Returns
    -------
    {float, np.ndarray}
        result of evaluation
    """
    return np.abs(4 * x - 2)


def roos_and_arnold_2_comparison(a_u: np.ndarray) -> Tuple[float, float]:
    """
    Calculates Roos and Arnold function 2 test integral of random samples.

    Parameters
    ----------
    a_u : numpy.ndarray
        random samples

    Returns
    -------
    actual : float
        test integral based on random samples
    expected : float
        analytic test integral
    """
    a_actual = roos_and_arnold_2(x=a_u)
    actual = np.mean(np.prod(a_actual, axis=1))
    expected = 0
    return actual, expected


def roos_and_arnold_3(x: float | np.ndarray) -> float | np.ndarray:
    """
    Evaluates Roos and Arnold function 3 test integral of random samples.

    Parameters
    ----------
    x : {float, np.ndarray}
        sample on which to evaluate

    Returns
    -------
    {float, np.ndarray}
        result of evaluation
    """
    return (np.pi / 2) * np.sin(np.pi * x / 2)


def roos_and_arnold_3_comparison(a_u: np.ndarray) -> Tuple[float, float]:
    """
    Calculates Roos and Arnold function 3 test integral of random samples.

    Parameters
    ----------
    a_u : numpy.ndarray
        random samples

    Returns
    -------
    actual : float
        test integral based on random samples
    expected : float
        analytic test integral
    """
    a_actual = roos_and_arnold_3(x=a_u)
    actual = np.mean(np.prod(a_actual, axis=1))
    expected = 1.0
    return actual, expected


def genzs_example(x: float | np.ndarray, aj_lst: List[float] | List[int], uj_lst: List[float] | List[int],
                  d: None | int = None) -> float | np.ndarray:
    """
    Evaluates Genzs example test integral of random samples.

    Parameters
    ----------
    x : {float, np.ndarray}
        sample on which to evaluate
    aj_lst : {List[float], List[int]}
        list of numbers, where each aj >= 0, that controls importance of each dimension
    uj_lst : {List[float], List[int]}
        list of random non-negative numbers (unrelated to u)
    d : {None, int}, optional
        number of dimensions

    Returns
    -------
    {float, np.ndarray}
        result of evaluation
    """
    aj = aj_lst if d is None else aj_lst[d]
    uj = uj_lst if d is None else uj_lst[d]
    return aj ** -2 + (x - uj) ** 2


def genzs_example_comparison(a_u: np.ndarray, aj_lst: List[float] | List[int],
                             uj_lst: List[float] | List[int]) -> Tuple[float, float]:
    """
    Calculates Genzs example test integral of random samples.

    Parameters
    ----------
    a_u : numpy.ndarray
        random samples
    aj_lst : {List[float], List[int]}
        list of numbers, where each aj >= 0, that controls importance of each dimension
    uj_lst : {List[float], List[int]}
        list of random non-negative numbers (unrelated to u)

    Returns
    -------
    actual : float
        test integral based on random samples
    expected : float
        analytic test integral
    """
    if len(aj_lst) != a_u.shape[1]:
        msg = (
            f"Invalid input, expected number of a_j(={len(aj_lst)}) values"
            f" to equal number of dimensions {a_u.shape[1]}."
        )
        raise ValueError(msg)
    invalid_aj_lst = [aj for aj in aj_lst if aj < 0]
    if len(invalid_aj_lst) > 0:
        msg = f"input contains invalid aj values: {invalid_aj_lst}"
        raise ValueError(msg)

    aj = np.array(aj_lst).reshape((1, len(aj_lst)))
    uj = np.array(uj_lst).reshape((1, len(uj_lst)))

    a_actual = genzs_example(x=a_u, aj_lst=aj, uj_lst=uj)
    actual = np.mean(np.prod(a_actual, axis=1))
    expected = np.prod(aj ** -2 + 1.0 / 3.0 - uj + uj ** 2)

    return actual, expected


def subcube_volume(x: float | np.ndarray, a: int = 0.5) -> float | np.ndarray:
    """
    Evaluates subcube volume test integral of random samples.

    Parameters
    ----------
    x : {float, np.ndarray}
        sample on which to evaluate
    a : int
        hyperparameter controlling relative importance of all u_j

    Returns
    -------
    {float, np.ndarray}
        result of evaluation
    """
    return (a - x) > 0


def subcube_volume_comparison(a_u: np.ndarray, a: int) -> Tuple[float, float]:
    """
    Calculates subcube volume test integral of random samples.

    Parameters
    ----------
    a_u : numpy.ndarray
        random samples
    a : int
        hyperparameter controlling relative importance of all u_j

    Returns
    -------
    actual : float
        test integral based on random samples
    expected : float
        analytic test integral
    """
    ndim = a_u.shape[1]
    a_actual = subcube_volume(x=a_u, a=a)
    a_actual = np.prod(a_actual + 1e-15, axis=1)

    actual = np.mean(a_actual)
    expected = (a - 0.5) ** ndim

    return actual, expected


def high_dim_1(x: float | np.ndarray, c: float = 0.01) -> float | np.ndarray:
    """
    Evaluates high dimensional function 1 test integral of random samples.

    Parameters
    ----------
    x : {float, np.ndarray}
        sample on which to evaluate
    c : float
        hyperparameter controlling relative importance of all u_j

    Returns
    -------
    {float, np.ndarray}
        result of evaluation
    """
    return 1 + c * (x - 0.5)


def high_dim_1_comparison(a_u: np.ndarray, c: float = 0.01) -> Tuple[float, float]:
    """
    Calculates high dimensional function 1 test integral of random samples.

    Parameters
    ----------
    a_u : numpy.ndarray
        random samples
    c : float
        hyperparameter controlling relative importance of all u_j

    Returns
    -------
    actual : float
        test integral based on random samples
    expected : float
        analytic test integral
    """
    a_actual = np.prod(high_dim_1(x=a_u, c=c), axis=1)
    actual = np.mean(a_actual)
    expected = 1

    return actual, expected


def high_dim_2(x: float | np.ndarray, d: int = 1, c0: float = 0.01, c: None | float = None) -> float | np.ndarray:
    """
    Evaluates high dimensional function 2 test integral of random samples.

    Parameters
    ----------
    x : {float, np.ndarray}
        sample on which to evaluate
    d : int
        dimensionality for which to evaluate sample
    c0 : float
        hyperparameter controlling relative importance of all u_j
    c : {None, float}, optional
        hyperparameter controlling relative importance of all u_j

    Returns
    -------
    {float, np.ndarray}
        result of evaluation
    """
    c = c0 / d if c is None else c
    return 1 + c * (x - 0.5)


def high_dim_2_comparison(a_u: np.ndarray, c0: float = 0.01) -> Tuple[float, float]:
    """
    Calculates high dimensional function 2 test integral of random samples.

    Parameters
    ----------
    a_u : numpy.ndarray
        random samples
    c0 : float
        hyperparameter controlling relative importance of all u_j

    Returns
    -------
    actual : float
        test integral based on random samples
    expected : float
        analytic test integral
    """
    c = c0 / (1 + np.arange(1, a_u.shape[1] + 1))
    a_actual = np.prod(high_dim_2(x=a_u, c=c), axis=1)
    actual = np.mean(a_actual)
    expected = 1
    return actual, expected


def high_dim_3(x: float | np.ndarray, d: int) -> float | np.ndarray:
    """
    Evaluates high dimensional function 3 test integral of random samples.

    Parameters
    ----------
    x : {float, np.ndarray}
        sample on which to evaluate
    d : int
        dimensionality for which to evaluate sample

    Returns
    -------
    {float, np.ndarray}
        result of evaluation
    """
    _lambda = np.sqrt(d / (1 + d))
    return x ** (_lambda - 1)


def high_dim_3_comparison(a_u: np.ndarray) -> Tuple[float, float]:
    """
    Calculates high dimensional function 3 test integral of random samples.

    Parameters
    ----------
    a_u : numpy.ndarray
        random samples

    Returns
    -------
    actual : float
        test integral based on random samples
    expected : float
        analytic test integral
    """
    a_dimensions = np.arange(1, a_u.shape[1] + 1)
    a_actual = np.prod(high_dim_3(x=a_u, d=a_dimensions), axis=1)

    actual = np.mean(a_actual)
    _lambda = np.sqrt(a_dimensions / (1 + a_dimensions))
    expected = np.product(1 / _lambda)

    return actual, expected


def high_dim_4(x: float | np.ndarray, d: int) -> float | np.ndarray:
    """
    Evaluates high dimensional function 4 test integral of random samples.

    Parameters
    ----------
    x : {float, np.ndarray}
        sample on which to evaluate
    d : int
        dimensionality for which to evaluate sample

    Returns
    -------
    {float, np.ndarray}
        result of evaluation
    """
    return (d - x) / (d - 0.5)


def high_dim_4_comparison(a_u: np.ndarray) -> Tuple[float, float]:
    """
    Calculates high dimensional function 4 test integral of random samples.

    Parameters
    ----------
    a_u : numpy.ndarray
        random samples

    Returns
    -------
    actual : float
        test integral based on random samples
    expected : float
        analytic test integral
    """
    ndim = a_u.shape[1]
    a_actual = np.prod(high_dim_4(x=a_u, d=ndim), axis=1)
    actual = np.mean(a_actual)
    expected = 1
    return actual, expected


def joe_kuo_1(x: float | np.ndarray, d: int) -> float | np.ndarray:
    """
    Evaluates Joe Kuo function 1 test integral of random samples.

    Parameters
    ----------
    x : {float, np.ndarray}
        sample on which to evaluate
    d : int
        dimensionality for which to evaluate sample

    Returns
    -------
    {float, np.ndarray}
        result of evaluation
    """
    c = d ** (1.0 / 3.0)
    return (np.abs(4 * x - 2) + c) / (1 + c)


def joe_kuo_1_comparison(a_u: np.ndarray) -> Tuple[float, float]:
    """
    Calculates Joe Kuo function 1 test integral of random samples.

    Parameters
    ----------
    a_u : numpy.ndarray
        random samples

    Returns
    -------
    actual : float
        test integral based on random samples
    expected : float
        analytic test integral
    """
    a_dimensions = np.arange(1, a_u.shape[1] + 1)
    a_actual = np.prod(joe_kuo_1(x=a_u, d=a_dimensions), axis=1)
    actual = np.mean(a_actual)
    expected = 1
    return actual, expected


def lds_investigations_f4(x: float | np.ndarray) -> float | np.ndarray:
    """
    Evaluates LDS investigations function 4 test integral of random samples.

    Parameters
    ----------
    x : {float, np.ndarray}
        sample on which to evaluate

    Returns
    -------
    {float, np.ndarray}
        result of evaluation
    """
    return np.sign(x - 0.5)


def lds_investigations_f4_comparison(a_u: np.ndarray) -> Tuple[float, float]:
    """
    Calculates LDS investigations function 4 test integral of random samples.

    Parameters
    ----------
    a_u : numpy.ndarray
        random samples

    Returns
    -------
    actual : float
        test integral based on random samples
    expected : float
        analytic test integral
    """
    x = lds_investigations_f4(x=a_u)  # if u<0.5, x = -1, otherwise x = 1
    f = np.prod(x, axis=1)
    actual = np.mean(f)
    expected = 0
    return actual, expected


def lds_investigations_f5(x: float | np.ndarray, a: int | float = 30, b: int | float = -15) -> float | np.ndarray:
    """
    Evaluates LDS investigations function 5 test integral of random samples.

    Parameters
    ----------
    x : {float, np.ndarray}
        sample on which to evaluate
    a : {int, float}
        scale parameter
    b : {int, float}
        intercept parameter

    Returns
    -------
    {float, np.ndarray}
        result of evaluation
    """
    v = np.exp(a * x + b)
    return (v - 1) / (v + 1)


def lds_investigations_f5_comparison(a_u: np.ndarray, a: int | float = 30, b: int | float = -15) -> Tuple[float, float]:
    """
    Calculates LDS investigations function 5 test integral of random samples.

    Parameters
    ----------
    a_u : numpy.ndarray
        random samples
    a : {int, float}
        scale parameter
    b : {int, float}
        intercept parameter

    Returns
    -------
    actual : float
        test integral based on random samples
    expected : float
        analytic test integral
    """
    a_actual = np.prod(lds_investigations_f5(x=a_u, a=a, b=b), axis=1)
    actual = np.mean(a_actual)
    expected = 0
    return actual, expected


def lds_investigations_f6(x: float | np.ndarray) -> float | np.ndarray:
    """
    Evaluates LDS investigations function 6 test integral of random samples.

    Parameters
    ----------
    x : {float, np.ndarray}
        sample on which to evaluate

    Returns
    -------
    {float, np.ndarray}
        result of evaluation
    """
    return -2.4 * np.sqrt(7) * (x - 0.5) + 8 * np.sqrt(7) * (x - 0.5) ** 3


def lds_investigations_f6_comparison(a_u: np.ndarray) -> Tuple[float, float]:
    """
    Calculates LDS investigations function 6 test integral of random samples.

    Parameters
    ----------
    a_u : numpy.ndarray
        random samples

    Returns
    -------
    actual : float
        test integral based on random samples
    expected : float
        analytic test integral
    """
    value = lds_investigations_f6(a_u)
    a_actual = np.prod(value, axis=1)
    actual = np.mean(a_actual)
    expected = 0
    return actual, expected


def lds_investigations_f7(x: float | np.ndarray) -> float | np.ndarray:
    """
    Evaluates LDS investigations function 7 test integral of random samples.

    Parameters
    ----------
    x : {float, np.ndarray}
        sample on which to evaluate

    Returns
    -------
    {float, np.ndarray}
        result of evaluation
    """
    return 2 * np.sqrt(3) * (x - 0.5)


def lds_investigations_f7_comparison(a_u: np.ndarray) -> Tuple[float, float]:
    """
    Calculates LDS investigations function 7 test integral of random samples.

    Parameters
    ----------
    a_u : numpy.ndarray
        random samples

    Returns
    -------
    actual : float
        test integral based on random samples
    expected : float
        analytic test integral
    """
    a_actual = np.prod(lds_investigations_f7(x=a_u), axis=1)
    actual = np.mean(a_actual)
    expected = 0
    return actual, expected


def optimization_f3(x: float | np.ndarray) -> float | np.ndarray:
    """
    Evaluates optimization function 3 test integral of random samples.

    Parameters
    ----------
    x : {float, np.ndarray}
        sample on which to evaluate

    Returns
    -------
    {float, np.ndarray}
        result of evaluation
    """
    return 4 * np.minimum(x, 1.0 - x)


def optimization_f3_comparison(a_u: np.ndarray) -> Tuple[float, float]:
    """
    Calculates optimization function 3 test integral of random samples.

    Parameters
    ----------
    a_u : numpy.ndarray
        random samples

    Returns
    -------
    actual : float
        test integral based on random samples
    expected : float
        analytic test integral
    """

    def row_func(x: float | np.ndarray) -> float:
        """
        Applies optimization function 3 on each sample.

        Parameters
        ----------
        x : {float, numpy.ndarray}
            random samples

        Returns
        -------
        float:
            output of optimization function 3
        """
        return np.prod(optimization_f3(x))

    actual = np.mean([row_func(a_u[i, :]) for i in range(a_u.shape[0])])
    expected = 1

    return actual, expected


def optimization_f4_analytic_integral_comparison(aj_lst: List[float] | List[int],
                                                 uj_lst: List[float] | List[int]) -> float:
    """
    Calculates analytic integral optimization function 4.

    Parameters
    ----
    aj_lst : {List(float), List(int)}
        list of numbers, where each aj >=} 0, that controls importance of each dimension
    uj_lst : {List(float), List(int)}
        list of numbers, where each uj ~ U(0, 1), that determine the x-intercepts of each dimension

    Returns
    -------
    float
        analytic test integral
    """

    def analytic_integral(aj, uj):
        return np.sqrt(np.pi) * (sp.erf(aj * uj) + sp.erf(aj - aj * uj)) / (2 * aj)

    return np.prod([analytic_integral(aj, uj) for aj, uj in zip(aj_lst, uj_lst)])


def optimization_f4_comparison(a_u: np.ndarray, aj_lst: List[float] | List[int],
                               uj_lst: List[float] | List[int]) -> Tuple[float, float]:
    """
    Calculates optimization function 4 test integral using random samples.

    Parameters
    ----------
    a_u  : numpy.ndarray
        random samples
    aj_lst : {List(float), List(int)}
        list of numbers, where each aj >= 0, that controls importance of each dimension
    uj_lst : {List(float), List(int)}
        list of numbers, where each uj ~ U(0, 1), that determine the x-intercepts of each dimension

    Returns
    -------
    actual : float
        test integral based on random samples
    expected : float
        analytic test integral

    Notes
    -----
    Results appear to be erroneous.
    """
    if len(aj_lst) != a_u.shape[1]:
        msg = (
            f"Invalid input, expected number of a_j(={len(aj_lst)}) values"
            f" to equal number of dimensions {a_u.shape[1]}."
        )
        raise ValueError(msg)
    invalid_aj_lst = [aj for aj in aj_lst if aj < 0]
    if len(invalid_aj_lst) > 0:
        msg = f"input contains invalid aj values: {invalid_aj_lst}"
        raise ValueError(msg)

    a = np.array(aj_lst).reshape(1, len(aj_lst))

    def row_func(a_x: np.ndarray, a_uj: np.ndarray) -> float:
        """
        Applies optimization function 4 on each sample.

        Parameters
        ----------
        a_x : numpy.ndarray
            random samples
        a_uj : numpy.ndarray
            number, where uj ~ U(0, 1), that determines the x-intercepts of each dimension

        Returns
        -------
        float:
            output of optimization function 4
        """
        return np.prod(np.exp(-a ** 2 * (a_x - a_uj) ** 2))

    actual = np.mean([row_func(a_u[j, :], a_uj) for j, a_uj in enumerate(uj_lst)])

    expected = optimization_f4_analytic_integral_comparison(aj_lst, uj_lst)
    return actual, expected


def optimization_f5_analytic_integral_comparison(aj_lst: List[float] | List[int],
                                                 uj_lst: List[float] | List[int]) -> float:
    """
    Calculates analytic integral optimization function 5.

    Parameters
    ----------
    aj_lst : {List(float), List(int)}
        list of numbers, where each aj >= 0, that controls importance of each dimension
    uj_lst : {List(float), List(int)}
        list of numbers, where each uj ~ U(0, 1), that determine the x-intercepts of each dimension

    Returns
    -------
    float
        analytic test integral
    """

    def analytic_integral(aj, uj):
        return np.exp(aj * uj) / aj - np.exp(-aj * (1 - uj)) / aj

    return np.prod([analytic_integral(aj, uj) for aj, uj in zip(aj_lst, uj_lst)])


def optimization_f5_comparison(a_u: np.ndarray, aj_lst: List[float] | List[int],
                               uj_lst: List[float] | List[int]) -> Tuple[float, float]:
    """
    Calculates optimization function 5 test integral of random samples.

    Parameters
    ----------
    a_u : numpy.ndarray
        random samples
    aj_lst : {List(float), List(int)}
        list of numbers, where each aj >= 0, that controls importance of each dimension
    uj_lst : {List(float), List(int)}
        list of numbers, where each uj ~ U(0, 1), that determine the x-intercepts of each dimension

    Returns
    -------
    actual : float
        test integral based on random samples
    expected : float
        analytic test integral

    Notes
    -----
    Results appear to be erroneous.
    """
    if len(aj_lst) != a_u.shape[1]:
        msg = (
            f"Invalid input, expected number of a_j(={len(aj_lst)}) values"
            f" to equal number of dimensions {a_u.shape[1]}."
        )
        raise ValueError(msg)
    invalid_aj_lst = [aj for aj in aj_lst if aj < 0]
    if len(invalid_aj_lst) > 0:
        msg = f"input contains invalid aj values: {invalid_aj_lst}"
        raise ValueError(msg)

    aj = np.array(aj_lst).reshape(1, len(aj_lst))

    def row_func(a_x: np.ndarray, uj: float) -> float:
        """
        Applies optimization function 4 on each sample.

        Parameters
        ----------
        a_x : numpy.ndarray
            random samples
        uj : float

        Returns
        -------
        float:
            output of optimization function 4
        """
        return np.prod(np.exp(-aj * np.abs(a_x - uj)))

    actual = np.mean([row_func(a_u[j, :], a_uj) for j, a_uj in enumerate(uj_lst)])
    expected = optimization_f5_analytic_integral_comparison(aj_lst, uj_lst)
    return actual, expected
