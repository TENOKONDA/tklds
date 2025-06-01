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

from tklds.integrals.additive import *
from tklds.constant import SequenceNum
from tklds.tests.utilities import compare_outputs

np.random.seed(1234)


class TestIntegralAdditive(unittest.TestCase):

    def test_roos_and_arnold_1_pcg64_numpoints_1000(self):
        compare_outputs(SequenceNum.PCG64, roos_and_arnold_1_comparison, places=1)

    def test_lds_investigations_f1_tkrg_a_ap5_numpoints_1000(self):
        compare_outputs(SequenceNum.TKRG_A_AP5, lds_investigations_f1_comparison, places=0)

    def test_lds_investigations_f2_new_joe_kuo_6_21201_numpoints_1000(self):
        compare_outputs(SequenceNum.NEW_JOE_KUO, lds_investigations_f2_comparison, places=0)

    def test_lds_investigations_f3_pcg64_numpoints_1000(self):
        compare_outputs(SequenceNum.PCG64, lds_investigations_f3_comparison, places=2)

    def test_lds_investigations_f8_tkrg_a_ap5_numpoints_1000(self):
        compare_outputs(SequenceNum.TKRG_A_AP5, lds_investigations_f8_comparison, places=1)

    def test_lds_investigations_f9_new_joe_kuo_6_21201_numpoints_1000(self):
        compare_outputs(SequenceNum.NEW_JOE_KUO, lds_investigations_f9_comparison, places=1)

    def test_optimization_f1_pcg64_numpoints_1000(self):
        compare_outputs(SequenceNum.PCG64, optimization_f1_comparison, places=2)

    def test_optimization_f2_tkrg_a_ap5_numpoints_1000(self):
        compare_outputs(SequenceNum.TKRG_A_AP5, optimization_f2_comparison, places=2)
