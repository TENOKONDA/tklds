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

import numpy.testing as npt
from tklds.constant import SequenceNum
from tklds.integrals.non_decomposable import _generate_points
from tklds.generators.constant import TKRG_A_AP5_FILEPATH, JOE_KUO_FILEPATH
from tklds.generators.iterative_lds import _load_dimensions_list_from_file, IterativeLDS
from tklds.tests.utilities import NUM_DIM, NUM_POINTS


class TestSobol(unittest.TestCase):

    def test_sobol_pcg64_numpoints_1000(self):
        _generate_points(SequenceNum.PCG64, num_points=NUM_POINTS, num_dim=NUM_DIM, num_skip_points=0)

    def test_sobol_tkrg_a_ap5_numpoints_1000(self):
        _generate_points(SequenceNum.TKRG_A_AP5, num_points=NUM_POINTS, num_dim=NUM_DIM, num_skip_points=0)

    def test_sobol_new_joe_kuo_6_21201_numpoints_1000(self):
        _generate_points(SequenceNum.NEW_JOE_KUO, num_points=NUM_POINTS, num_dim=NUM_DIM, num_skip_points=0)

    def test_iterative_lds_tkrg_a_ap5_numpoints_1000(self):
        dimensions_lst = _load_dimensions_list_from_file(TKRG_A_AP5_FILEPATH, drop_implicit=False)[:NUM_DIM]
        lds_iter = IterativeLDS(dimensions_lst)
        u_iter = lds_iter.rvs((NUM_POINTS, NUM_DIM))
        u_lds = _generate_points(SequenceNum.TKRG_A_AP5, num_points=NUM_POINTS, num_dim=NUM_DIM, num_skip_points=0)
        npt.assert_almost_equal(u_lds, u_iter, decimal=1)

    def test_iterative_lds_new_joe_kuo_6_21201_numpoints_1000(self):
        dimensions_lst = _load_dimensions_list_from_file(JOE_KUO_FILEPATH, drop_implicit=False)[:NUM_DIM]
        lds_iter = IterativeLDS(dimensions_lst)
        u_iter = lds_iter.rvs((NUM_POINTS, NUM_DIM))
        u_lds = _generate_points(SequenceNum.NEW_JOE_KUO, num_points=NUM_POINTS, num_dim=NUM_DIM, num_skip_points=0)
        npt.assert_equal(u_lds, u_iter)

    def test_iterative_lds_new_joe_kuo_6_21201_numpoints_1000_fast_loading(self):
        dimensions_lst = _load_dimensions_list_from_file(JOE_KUO_FILEPATH, drop_implicit=False,
                                                         fast_loading=False)[:NUM_DIM]
        lds_iter = IterativeLDS(dimensions_lst)
        u_iter = lds_iter.rvs((NUM_POINTS, NUM_DIM))
        u_lds = _generate_points(SequenceNum.NEW_JOE_KUO, num_points=NUM_POINTS, num_dim=NUM_DIM, num_skip_points=0)
        npt.assert_equal(u_lds, u_iter)

    def test_iterative_lds_new_joe_kuo_6_21201_numpoints_1000_no_dim_defined(self):
        dimensions_lst = _load_dimensions_list_from_file(JOE_KUO_FILEPATH, drop_implicit=False)[:NUM_DIM]
        lds_iter = IterativeLDS(dimensions_lst)
        u_iter = lds_iter.rvs((NUM_POINTS,))
        u_lds = _generate_points(SequenceNum.NEW_JOE_KUO, num_points=NUM_POINTS, num_dim=NUM_DIM, num_skip_points=0)
        npt.assert_equal(u_lds, u_iter)

    def test_iterative_lds_new_joe_kuo_6_21201_numpoints_1000_value_error_invalid_number_dimensions(self):
        dimensions_lst = _load_dimensions_list_from_file(JOE_KUO_FILEPATH, drop_implicit=False)[:NUM_DIM]
        with self.assertRaises(ValueError):
            try:
                lds_iter = IterativeLDS(dimensions_lst)
                lds_iter.rvs((NUM_POINTS, 24))
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = "Invalid number of dimensions requested"
        self.assertIn(expected_error, error_str)

    def test_iterative_lds_new_joe_kuo_6_21201_numpoints_1000_value_error_invalid_size(self):
        dimensions_lst = _load_dimensions_list_from_file(JOE_KUO_FILEPATH, drop_implicit=False)[:NUM_DIM]
        with self.assertRaises(ValueError):
            try:
                lds_iter = IterativeLDS(dimensions_lst)
                lds_iter.rvs((NUM_POINTS, NUM_DIM, 24))
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = "Invalid size specified: 3, must either be an int, one-tuple or two-tuple"
        self.assertIn(expected_error, error_str)
