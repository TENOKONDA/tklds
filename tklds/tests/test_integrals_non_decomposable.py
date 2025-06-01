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

import pandas as pd

from tklds.integrals.non_decomposable import *
from tklds.integrals.non_decomposable import _generate_points, _improper_integral_benchmark
from tklds.tests.utilities import compare_outputs, NUM_DIM

np.random.seed(1234)


class TestIntegralNonDecomposable(unittest.TestCase):

    def test_generate_points_type_error_non_sequencenum(self):
        sequence = "trkg"
        with self.assertRaises(TypeError):
            try:
                _generate_points(sequence, num_points=5, num_dim=5, num_skip_points=1)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Expected 'sequence' to be a SequenceNum type, received: {type(sequence).__name__}"
        self.assertIn(expected_error, error_str)

    def test_genzs_discontinuous_function_pcg64_numpoints_1000(self):
        mu, a = 0.5, 0.5
        function = partial(genzs_discontinuous_function, mu=mu, a=a, d=NUM_DIM)
        compare_outputs(SequenceNum.PCG64, function, places=1)

    def test_atanassov_function_tkrg_a_ap5_numpoints_1000(self):
        function = partial(atanassov_function, d=NUM_DIM)
        compare_outputs(SequenceNum.TKRG_A_AP5, function, places=3)

    def test_improper_integral_benchmark(self):
        actual = _improper_integral_benchmark(num_points_lst=[2], d=5, verbose=False).astype("object")
        expected = pd.DataFrame(
            {
                "generator": ["tkrg-a-ap5", "new-joe-kuo", "pcg64"],
                "num_samples": [2, 2, 2],
                "c": [0.031250, 0.031250, 0.03102867171540123]
            }
        ).astype("object")
        pd.testing.assert_frame_equal(actual, expected)
