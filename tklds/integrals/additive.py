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

from typing import Tuple

import numpy as np


def roos_and_arnold_1(x: float | np.ndarray, d: int = 1) -> float | np.ndarray:
    """
    Evaluates the Roos and Arnold 1 function.

    Parameters
    ----------
    x : {float, np.ndarray}
        sample on which to evaluate
    d : int
        dimensionality for which to evaluate sample

    Returns
    -------
    {float, np.ndarray]
        result of evaluation
    """
    return (1 / d) * (4 * x - 2)


def roos_and_arnold_1_comparison(a_u: np.ndarray) -> Tuple[float, float]:
    """
    Calculates Roos and Arnold function 1 test integral of random samples.

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
    d = a_u.shape[1]
    actual = np.mean(np.sum(roos_and_arnold_1(a_u, d=d), axis=1))
    expected = 0
    return actual, expected


def lds_investigations_f1(x: float | np.ndarray, d: int) -> float | np.ndarray:
    """
    Evaluates LDS investigations function 1.

    Parameters
    ----------
    x : {float, np.ndarray}
        sample on which to evaluate
    d : int
        dimensionality for which to evaluate sample

    Returns
    -------
    {float, np.ndarray]
        result of evaluation
    """
    return np.sqrt(12.0 / d) * (x - 0.5)


def lds_investigations_f1_comparison(a_u: np.ndarray) -> Tuple[float, float]:
    """
    Calculates LDS investigations function 1 test integral of random samples.

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
    d = a_u.shape[1]
    a_actual = np.sum(lds_investigations_f1(a_u, d), axis=1)
    actual = np.mean(a_actual)
    expected = 0
    return actual, expected


def lds_investigations_f2(x: float | np.ndarray, d: int) -> float | np.ndarray:
    """
    Evaluates LDS investigations function 2.

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
    return np.sqrt(45.0 / (4 * d)) * (x ** 2 - 1 / 3.0)


def lds_investigations_f2_comparison(a_u: np.ndarray) -> Tuple[float, float]:
    """
    Calculates LDS investigations function 2 test integral of random samples.

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
    d = a_u.shape[1]
    a_x = np.sum(lds_investigations_f2(a_u, d=d), axis=1)
    actual = np.mean(a_x)
    expected = 0
    return actual, expected


def lds_investigations_f3(x: float | np.ndarray, d: int = 1) -> float | np.ndarray:
    """
    Evaluates LDS investigations function 3.

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
    return np.sqrt(18 / d) * (x ** 0.5 - (2 / 3.0))


def lds_investigations_f3_comparison(a_u: np.ndarray) -> Tuple[float, float]:
    """
    Calculates LDS investigations function 3 test integral of random samples.

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
    d = a_u.shape[1]
    x = np.sum(lds_investigations_f3(a_u, d), axis=1)
    actual = np.mean(x)
    expected = 0
    return actual, expected


def lds_investigations_f8(x: float | np.ndarray, d: int) -> float | np.ndarray:
    """
    Evaluates the LDS investigations f8 function.

    Parameters
    ----------
    x : {float, np.ndarray}
        sample on which to evaluate
    d : int
        dimensionality for which to evaluate sample

    Returns
    -------
    {float, np.ndarray}
        output of LDS investigations function 9
    """

    def fvalue(x: float) -> int:
        """"
        Calculates f-value of random number (non-continuous function).

        Parameters
        ----------
        x : float
            random number

        Returns
        -------
        int
            f-value of x
        """
        if x < 1.0 / 6 or x > 4.0 / 6:
            return 1.0
        elif x == 1.0 / 6 or x == 4.0 / 6:
            return 0
        else:
            return -1.0

    def row_function(a_row: np.ndarray) -> int:
        """
        Performs nested summation of f-values for a given sample, i.e. sum_j sum_{i<j} fi * fj.

        Parameters
        ----------
        a_row : numpy.ndarray
            f-values of each dimension for a given sample

        Returns
        -------
        int
            result of nested summation of f-values
        """
        terms = [np.sum(a_row[:j] * a_row[j]) for j in range(len(a_row))]
        return np.sum(terms)

    sf = np.sqrt(2 / (d * (d - 1)))

    f = np.vectorize(fvalue)(x)
    return sf * row_function(f)


def lds_investigations_f8_comparison(a_u: np.ndarray) -> Tuple[float, float]:
    """
    Calculates LDS investigations function 8 test integral of random samples.

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
    d = a_u.shape[1]

    expected = 0
    actual = np.mean([lds_investigations_f8(a_u[i, :], d) for i in range(a_u.shape[0])])
    return actual, expected


def lds_investigations_f9(x: float | np.ndarray, d: int) -> float | np.ndarray:
    """
    Evaluates LDS investigations function 9.

    Parameters
    ----------
    x : float | np.ndarray
        sample on which to evaluate
    d : int
        dimensionality for which to evaluate sample

    Returns
    -------
    {float, np.ndarray}
        output of LDS investigations function 9.
    """

    def row_function(a_row: np.ndarray) -> int:
        """
        Performs nested summation of f-values for a given sample, i.e. sum_j sum_{i<j} fi * fj.

        Parameters
        ----------
        a_row : numpy.ndarray
            f-values of each dimension for a given sample

        Returns
        -------
        int
            result of nested summation of f-values
        """
        terms = [np.sum(a_row[:j] * a_row[j]) for j in range(len(a_row))]
        return np.sum(terms)

    sf = np.sqrt(2 / (d * (d - 1)))
    f = 27.20917094 * (x ** 3) - 36.19250850 * (x ** 2) + 8.983337562 * x + 0.7702079855
    return sf * row_function(f)


def lds_investigations_f9_comparison(a_u: np.ndarray) -> Tuple[float, float]:
    """
    Calculates LDS investigations function 9 test integral of random samples.

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
    d = a_u.shape[1]
    expected = 0
    actual = np.mean([lds_investigations_f9(a_u[i, :], d) for i in range(a_u.shape[0])])
    return actual, expected


def optimization_f1(x: float | np.ndarray, d: int = 1) -> float:
    """
    Applies optimization function 1 on each sample.

    Parameters
    ----------
    x : {float, numpy.ndarray}
        random samples
    d : int
        dimensionality for which to evaluate sample

    Returns
    -------
    float:
        output of optimization function 1
    """
    x1 = x[:-1]
    x2 = x[1:]
    return np.sum(x1 * x2) * (4 / (d - 1))


def optimization_f1_comparison(a_u: np.ndarray) -> Tuple[float, float]:
    """
    Calculates optimization function 1 test integral of random samples.

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
    d = a_u.shape[1]
    actual = np.mean([optimization_f1(a_u[i, :], d) for i in range(a_u.shape[0])])
    expected = 1

    return actual, expected


def optimization_f2(x: float | np.ndarray, d: int = 1) -> float:
    """
    Applies optimization function 2 on each sample.

    Parameters
    ----------
    x : {float, numpy.ndarray}
        random samples
    d : int
        dimensionality for which to evaluate sample

    Returns
    -------
    float:
        output of optimization function 2
    """
    x1 = x[:-2]
    x2 = x[1:-1]
    x3 = x[2:]
    return np.sum(x1 * x2 * x3) * (8 / (d - 2))


def optimization_f2_comparison(a_u: np.ndarray) -> Tuple[float, float]:
    """
    Calculates optimization function 2 test integral of random samples.

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
    d = a_u.shape[1]
    actual = np.mean([optimization_f2(a_u[i, :], d) for i in range(a_u.shape[0])])
    expected = 1
    return actual, expected
