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
import traceback
from functools import partial
from tklds.constant import SequenceNum
from tklds.integrals.multiplicative import *
from tklds.tests.utilities import compare_outputs, NUM_DIM, get_actual_expected_outputs


np.random.seed(1234)


class TestIntegralMultiplicative(unittest.TestCase):
    def test_hellekalek_function_pcg64_numpoints_1000(self):
        function = partial(hellekalek_function_comparison, alpha=1)
        compare_outputs(SequenceNum.PCG64, function, places=10)

    def test_hellekalek_function_pcg64_numpoints_100_value_error_alpha_low(self):
        alpha = 0
        function = partial(hellekalek_function_comparison, alpha=alpha)
        with self.assertRaises(ValueError):
            try:
                get_actual_expected_outputs(SequenceNum.PCG64, function)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e

        expected_error = f"input alpha values outside range [1,3] with value: {alpha}"
        self.assertIn(expected_error, error_str)

    def test_hellekalek_function_pcg64_numpoints_100_value_error_alpha_high(self):
        alpha = 4
        function = partial(hellekalek_function_comparison, alpha=alpha)
        with self.assertRaises(ValueError):
            try:
                get_actual_expected_outputs(SequenceNum.PCG64, function)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e

        expected_error = f"input alpha values outside range [1,3] with value: {alpha}"
        self.assertIn(expected_error, error_str)

    def test_sobol_1_tkrg_a_ap5_numpoints_1000(self):
        aj_lst = np.arange(1, NUM_DIM + 1)
        function = partial(sobol_1_comparison, aj_lst=aj_lst)
        compare_outputs(SequenceNum.TKRG_A_AP5, function, places=1)

    def test_sobol_1_tkrg_a_ap5_numpoints_1000_value_error_wrong_shape_aj_lst(self):
        aj_lst = np.arange(1, NUM_DIM + 5)
        function = partial(sobol_1_comparison, aj_lst=aj_lst)
        with self.assertRaises(ValueError):
            try:
                get_actual_expected_outputs(SequenceNum.TKRG_A_AP5, function)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e

        expected_error = f"Invalid input, expected number of a_j(={len(aj_lst)}) values " \
                         f"to equal number of dimensions {NUM_DIM}."
        self.assertIn(expected_error, error_str)

    def test_sobol_1_tkrg_a_ap5_numpoints_1000_value_error_invalid_elements_aj_lst(self):
        aj_lst = np.arange(1, NUM_DIM + 1)
        aj_lst[2] = -1
        function = partial(sobol_1_comparison, aj_lst=aj_lst)
        with self.assertRaises(ValueError):
            try:
                get_actual_expected_outputs(SequenceNum.TKRG_A_AP5, function)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e

        expected_error = f"input contains invalid aj values: [-1]"
        self.assertIn(expected_error, error_str)

    def test_sobol_2_new_joe_kuo_6_21201_numpoints_1000(self):
        compare_outputs(SequenceNum.NEW_JOE_KUO, sobol_2_comparison, places=2)

    def test_owens_example_pcg64_numpoints_1000(self):
        compare_outputs(SequenceNum.PCG64, owens_example_comparison, places=-9)

    def test_roos_and_arnold_2_tkrg_a_ap5_numpoints_1000(self):
        compare_outputs(SequenceNum.TKRG_A_AP5, roos_and_arnold_2_comparison, places=1)

    def test_roos_and_arnold_3_new_joe_kuo_6_21201_numpoints_1000(self):
        compare_outputs(SequenceNum.NEW_JOE_KUO, roos_and_arnold_3_comparison, places=-7)

    def test_genzs_example_pcg64_numpoints_1000(self):
        aj_lst = np.arange(1, NUM_DIM + 1, dtype=float)
        uj_lst = np.arange(1, NUM_DIM + 1, dtype=float)
        function = partial(genzs_example_comparison, aj_lst=aj_lst, uj_lst=uj_lst)
        compare_outputs(SequenceNum.PCG64, function, places=-126)

    def test_genzs_example_pcg64_numpoints_1000_value_error_wrong_shape_aj_lst(self):
        aj_lst = np.arange(1, NUM_DIM + 5, dtype=float)
        uj_lst = np.arange(1, NUM_DIM + 1, dtype=float)
        function = partial(genzs_example_comparison, aj_lst=aj_lst, uj_lst=uj_lst)
        with self.assertRaises(ValueError):
            try:
                get_actual_expected_outputs(SequenceNum.PCG64, function)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e

        expected_error = f"Invalid input, expected number of a_j(={len(aj_lst)}) values " \
                         f"to equal number of dimensions {NUM_DIM}."
        self.assertIn(expected_error, error_str)

    def test_genzs_example_pcg64_numpoints_1000_value_error_invalid_values_aj_lst(self):
        aj_lst = np.arange(1, NUM_DIM + 1, dtype=float)
        aj_lst[1] = -1
        uj_lst = np.arange(1, NUM_DIM + 1, dtype=float)
        function = partial(genzs_example_comparison, aj_lst=aj_lst, uj_lst=uj_lst)
        with self.assertRaises(ValueError):
            try:
                get_actual_expected_outputs(SequenceNum.PCG64, function)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e

        expected_error = f"input contains invalid aj values: [-1.0]"
        self.assertIn(expected_error, error_str)

    def test_subcube_volume_tkrg_a_ap5_numpoints_1000(self):
        function = partial(subcube_volume_comparison, a=0.5)
        compare_outputs(SequenceNum.TKRG_A_AP5, function, places=2)

    def test_high_dim_1_new_joe_kuo_6_21201_numpoints_1000(self):
        function = partial(high_dim_1_comparison, c=0.01)
        compare_outputs(SequenceNum.NEW_JOE_KUO, function, places=3)

    def test_high_dim_2_pcg64_numpoints_1000(self):
        function = partial(high_dim_2_comparison, c0=0.01)
        compare_outputs(SequenceNum.PCG64, function, places=3)

    def test_high_dim_3_tkrg_a_ap5_numpoints_1000(self):
        compare_outputs(SequenceNum.TKRG_A_AP5, high_dim_3_comparison, places=-1)

    def test_high_dim_4_new_joe_kuo_6_21201_numpoints_1000(self):
        compare_outputs(SequenceNum.NEW_JOE_KUO, high_dim_4_comparison, places=3)

    def test_joe_kuo_1_pcg64_numpoints_1000(self):
        compare_outputs(SequenceNum.PCG64, joe_kuo_1_comparison, places=1)

    def test_lds_investigations_f4_tkrg_a_ap5_numpoints_1000(self):
        compare_outputs(SequenceNum.TKRG_A_AP5, lds_investigations_f4_comparison, places=2)

    def test_lds_investigations_f5_new_joe_kuo_6_21201_numpoints_1000(self):
        a = 30
        b = -15
        function = partial(lds_investigations_f5_comparison, a=a, b=b)
        compare_outputs(SequenceNum.NEW_JOE_KUO, function, places=2)

    def test_lds_investigations_f6_pcg64_numpoints_1000(self):
        compare_outputs(SequenceNum.PCG64, lds_investigations_f6_comparison, places=1)

    def test_lds_investigations_f7_tkrg_a_ap5_numpoints_1000(self):
        compare_outputs(SequenceNum.TKRG_A_AP5, lds_investigations_f7_comparison, places=5)

    def test_optimization_f3_new_joe_kuo_6_21201_numpoints_1000(self):
        compare_outputs(SequenceNum.NEW_JOE_KUO, optimization_f3_comparison, places=-13)
