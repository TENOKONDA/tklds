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

import unittest
from functools import partial

import numpy.testing as npt
from tklds.integrals.effective_dimensions import *
from tklds.tests.utilities import NUM_DIM


def g_hellakalek(x, alpha):
    return (x ** alpha - 1) / (1 + alpha)


def g_sobol_2(x, j):
    return (j + 2 * x) / (1 + j)


def make_g_lst_sobol_2(d):
    return [partial(g_sobol_2, j=sub_d) for sub_d in range(1, d + 1)]


class TestEffectiveDimensions(unittest.TestCase):
    def test_mean_dimension_additive_superposition(self):
        g_list = [1, 2, 3, 4]
        actual = mean_dimension_additive_superposition(g_list)
        expected = 1
        self.assertEqual(expected, actual)

    def test_mean_dimension_additive_truncation(self):
        g_list = [partial(g_hellakalek, alpha=1)] * NUM_DIM
        actual = mean_dimension_additive_truncation(g_list)
        expected = 25.5
        self.assertAlmostEqual(expected, actual, places=5)

    def test_mean_dimension_multiplicative_superposition(self):
        actual = mean_dimension_multiplicative_superposition(make_g_lst_sobol_2(NUM_DIM))
        expected = 1.082900
        self.assertAlmostEqual(expected, actual, places=6)

    def test_get_dimension_type(self):
        self.assertEqual(get_dimension_type(EffectiveDimensionNum.ADDITIVE_SUPERPOSITION),
                         mean_dimension_additive_superposition)
        self.assertEqual(get_dimension_type(EffectiveDimensionNum.ADDITIVE_TRUNCATION),
                         mean_dimension_additive_truncation)
        self.assertEqual(get_dimension_type(EffectiveDimensionNum.MULTIPLICATIVE_SUPERPOSITION),
                         mean_dimension_multiplicative_superposition)

    def test_mean_effective_dimension_single(self):
        function = partial(g_hellakalek, alpha=1)
        a_dimensions = np.arange(1, 5, 1)
        actual = mean_effective_dimension_single(function, EffectiveDimensionNum.MULTIPLICATIVE_SUPERPOSITION,
                                                 dimensions=a_dimensions)
        actual = np.array(actual)
        expected = np.array([1., 1.14286, 1.2973, 1.46286])
        npt.assert_almost_equal(expected, actual, decimal=5)
