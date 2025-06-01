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
import numpy as np
from tklds.constant import SequenceNum
from tklds.generators.iterative_lds import _load_iterative_lds_tkrg_a_ap5, IterativeLDS
from tklds.generators.sobol_engine import SobolEngine
from tklds.interface.generators import generate_lds_rvs, create_iterative_lds_generator, create_sobol_lds_engine, \
    get_sequence_enum


class TestInterfaceGenerators(unittest.TestCase):

    def test_get_lds_rvs(self):
        d = 10
        n = 100
        u_actual = generate_lds_rvs(sequence=SequenceNum.TKRG_A_AP5, n=n, d=d)
        iterative_lds = _load_iterative_lds_tkrg_a_ap5(d=d)
        u_expected = iterative_lds.rvs(n)
        self.assertTrue(np.all(u_actual == u_expected))

    def test_generate_lds_rvs_SequenceNum_type_error(self):
        sequence = "tkrg"
        with self.assertRaises(TypeError):
            try:
                generate_lds_rvs(sequence=sequence, n=2, d=2)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Expected 'sequence' to be an SequenceNum type, received: EnumMeta"
        self.assertIn(expected_error, error_str)

    def test_generate_lds_rvs_n_type_error(self):
        n = "2"
        with self.assertRaises(TypeError):
            try:
                generate_lds_rvs(sequence=SequenceNum.TKRG_A_AP5, n=n, d=2)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Expected 'n' to be an integer type, received: {type(n).__name__}"
        self.assertIn(expected_error, error_str)

    def test_generate_lds_rvs_d_type_error(self):
        d = "2"
        with self.assertRaises(TypeError):
            try:
                generate_lds_rvs(sequence=SequenceNum.TKRG_A_AP5, n=2, d=d)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Expected 'd' to be an integer type, received: {type(d).__name__}"
        self.assertIn(expected_error, error_str)

    def test_generate_lds_rvs_n_value_error_low(self):
        n = 0
        with self.assertRaises(ValueError):
            try:
                generate_lds_rvs(sequence=SequenceNum.TKRG_A_AP5, n=n, d=2)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Invalid value {n=}, must be in range [1, 2**32]."
        self.assertIn(expected_error, error_str)

    def test_generate_lds_rvs_n_value_error_high(self):
        n = 2**33
        with self.assertRaises(ValueError):
            try:
                generate_lds_rvs(sequence=SequenceNum.TKRG_A_AP5, n=n, d=2)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Invalid value {n=}, must be in range [1, 2**32]."
        self.assertIn(expected_error, error_str)

    def test_generate_lds_rvs_d_value_error(self):
        d = 0
        with self.assertRaises(ValueError):
            try:
                generate_lds_rvs(sequence=SequenceNum.TKRG_A_AP5, n=4, d=d)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Invalid value {d=}, must be greater than or equal to 1."
        self.assertIn(expected_error, error_str)

    def test_get_iterative_lds_generator(self):
        iterative_lds = create_iterative_lds_generator(sequence=SequenceNum.NEW_JOE_KUO, d=123)
        self.assertIsInstance(iterative_lds, IterativeLDS)
        self.assertEqual(123, iterative_lds.ndim)
        self.assertEqual((123, 34), iterative_lds.a_v.shape)
        self.assertEqual(0, iterative_lds.current_num_samples)
        self.assertTrue(np.all(iterative_lds.a_current_sobol_ints == np.zeros(123, dtype=np.uintc)))

    def test_get_iterative_lds_generator_max_points_less_than_d(self):
        iterative_lds = create_iterative_lds_generator(sequence=SequenceNum.NEW_JOE_KUO, d=123, max_points=3)
        self.assertIsInstance(iterative_lds, IterativeLDS)
        self.assertEqual(123, iterative_lds.ndim)
        self.assertEqual((123, 4), iterative_lds.a_v.shape)
        self.assertEqual(0, iterative_lds.current_num_samples)
        self.assertTrue(np.all(iterative_lds.a_current_sobol_ints == np.zeros(123, dtype=np.uintc)))

    def test_get_iterative_lds_generator_SequenceNum_type_error(self):
        sequence = "tkrg"
        with self.assertRaises(TypeError):
            try:
                create_iterative_lds_generator(sequence=sequence, d=123)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Expected 'sequence' to be an SequenceNum type, received: str"
        self.assertIn(expected_error, error_str)

    def test_get_iterative_lds_generator_d_type_error(self):
        d = "4"
        with self.assertRaises(TypeError):
            try:
                create_iterative_lds_generator(sequence=SequenceNum.TKRG_A_AP5, d=d)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Expected 'd' to be an integer type, received: {type(d).__name__}"
        self.assertIn(expected_error, error_str)

    def test_get_iterative_lds_generator_max_points_type_error(self):
        max_points = "4"
        with self.assertRaises(TypeError):
            try:
                create_iterative_lds_generator(sequence=SequenceNum.TKRG_A_AP5, d=6, max_points=max_points)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Expected 'max_points' to be an integer type, received: {type(max_points).__name__}"
        self.assertIn(expected_error, error_str)

    def test_get_iterative_lds_generator_d_value_error(self):
        d = 0
        with self.assertRaises(ValueError):
            try:
                create_iterative_lds_generator(sequence=SequenceNum.TKRG_A_AP5, d=d)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Invalid value {d=}, must be greater than or equal to 1."
        self.assertIn(expected_error, error_str)

    def test_get_iterative_lds_generator_max_points_value_error_low(self):
        max_points = 0
        with self.assertRaises(ValueError):
            try:
                create_iterative_lds_generator(sequence=SequenceNum.TKRG_A_AP5, d=4, max_points=max_points)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Invalid value {max_points=}, must be in range [1, 2**32]."
        self.assertIn(expected_error, error_str)

    def test_get_iterative_lds_generator_max_points_value_error_high(self):
        max_points = 2**33
        with self.assertRaises(ValueError):
            try:
                create_iterative_lds_generator(sequence=SequenceNum.TKRG_A_AP5, d=4, max_points=max_points)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Invalid value {max_points=}, must be in range [1, 2**32]."
        self.assertIn(expected_error, error_str)

    def test_get_sobol_lds_engine(self):
        d = 102
        sobol_engine = create_sobol_lds_engine(sequence=SequenceNum.TKRG_A_AP5, d=d)
        self.assertIsInstance(sobol_engine, SobolEngine)
        self.assertEqual(d, sobol_engine.d)

    def test_get_sobol_lds_engine_SequenceNum_type_error(self):
        sequence = "tkrg"
        with self.assertRaises(TypeError):
            try:
                create_sobol_lds_engine(sequence=sequence, d=123)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Expected 'sequence' to be an SequenceNum type, received: str"
        self.assertIn(expected_error, error_str)

    def test_get_sobol_lds_engine_d_type_error(self):
        d = "4"
        with self.assertRaises(TypeError):
            try:
                create_sobol_lds_engine(sequence=SequenceNum.TKRG_A_AP5, d=d)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Expected 'd' to be an integer type, received: {type(d).__name__}"
        self.assertIn(expected_error, error_str)

    def test_get_sobol_lds_engine_scramble_type_error(self):
        scramble = 1
        with self.assertRaises(TypeError):
            try:
                create_sobol_lds_engine(sequence=SequenceNum.TKRG_A_AP5, d=123, scramble=scramble)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Expected 'scramble' to be a boolean type, received: {type(scramble).__name__}."
        self.assertIn(expected_error, error_str)

    def test_get_sobol_lds_engine_d_value_error(self):
        d = 0
        with self.assertRaises(ValueError):
            try:
                create_sobol_lds_engine(sequence=SequenceNum.TKRG_A_AP5, d=d)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Invalid value {d=}, must be greater than or equal to 1."
        self.assertIn(expected_error, error_str)

    def test_get_sequence_enum(self):
        self.assertEqual(get_sequence_enum(), SequenceNum)
