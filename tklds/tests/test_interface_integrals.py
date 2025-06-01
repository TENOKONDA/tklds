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
from tklds.integrals.additive import roos_and_arnold_1
from tklds.integrals.multiplicative import high_dim_4
from tklds.integrals.non_decomposable import atanassov_function
from tklds.integrals.effective_dimensions import *
from tklds.interface.integrals import *


class TestInterfaceIntegrals(unittest.TestCase):

    def test_get_integral_by_name_additive(self):
        integral = get_integral_by_name("roos_and_arnold_1")
        self.assertEqual(integral, roos_and_arnold_1)

    def test_get_integral_by_name_multiplicative(self):
        integral = get_integral_by_name("high_dim_4")
        self.assertEqual(integral, high_dim_4)

    def test_get_integral_by_name_atanassov_function(self):
        integral = get_integral_by_name("atanassov_function")
        self.assertEqual(integral, atanassov_function)

    def test_get_integral_by_name_invalid_name(self):
        invalid_integral_name = "invalid_integral"
        with self.assertRaises(ValueError):
            try:
                get_integral_by_name(invalid_integral_name)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Integral with integral_name='{invalid_integral_name}' not found!"
        self.assertIn(expected_error, error_str)

    def test_get_integral_effective_dimension_function_additive_superposition(self):
        effective_dimension_fn = get_integral_effective_dimension_function(EffectiveDimensionNum.ADDITIVE_SUPERPOSITION)
        self.assertEqual(effective_dimension_fn, mean_dimension_additive_superposition)

    def test_get_integral_effective_dimension_function_additive_truncation(self):
        effective_dimension_fn = get_integral_effective_dimension_function(EffectiveDimensionNum.ADDITIVE_TRUNCATION)
        self.assertEqual(effective_dimension_fn, mean_dimension_additive_truncation)

    def test_get_integral_effective_dimension_function_multiplicative_superposition(self):
        effective_dimension_fn = get_integral_effective_dimension_function(
            EffectiveDimensionNum.MULTIPLICATIVE_SUPERPOSITION)
        self.assertEqual(effective_dimension_fn, mean_dimension_multiplicative_superposition)

    def test_get_integral_effective_dimension_function_invalid_type(self):
        invalid_type = "additive_superposition"
        with self.assertRaises(TypeError):
            try:
                get_integral_effective_dimension_function(invalid_type)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Expected 'effective_dimension' to be an SequenceNum type, received: str"
        self.assertIn(expected_error, error_str)

    def test_get_sequence_enum(self):
        self.assertEqual(SequenceNum, get_sequence_enum())

    def test_get_effective_dimension_enum(self):
        self.assertEqual(EffectiveDimensionNum, get_effective_dimension_enum())
