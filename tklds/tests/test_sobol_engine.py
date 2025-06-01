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

import numpy as np
from scipy.stats.qmc import Sobol

from tklds.constant import SequenceNum
from tklds.generators.iterative_lds import _load_iterative_lds_tkrg_a_ap5
from tklds.generators.sobol_engine import SobolEngine


class TestSobolEngine(unittest.TestCase):

    def test_sobol_engine_vs_iterative_lds(self):
        d = 10
        n = 123
        sobol_engine = SobolEngine(d=d, scramble=False)

        iterative_lds = _load_iterative_lds_tkrg_a_ap5(d=d)

        sobol_engine_rvs = sobol_engine.random(n)
        iterative_lds_rvs = iterative_lds.rvs((n, d))

        elementwise_equal = sobol_engine_rvs == iterative_lds_rvs

        self.assertTrue(np.all(elementwise_equal))

    def test_sobol_engine_fast_forward(self):
        d = 162
        n = 52
        sobol_engine = SobolEngine(d=d, scramble=False)
        sobol_engine_ff = SobolEngine(d=d, scramble=False)
        sobol_engine_ff = sobol_engine_ff.fast_forward(n=1)

        rvs = sobol_engine.random(n=n)
        rvs_ff = sobol_engine_ff.random(n=n)

        self.assertTrue(np.all(rvs[0, :] == np.zeros(d, dtype=float)))
        self.assertTrue(np.all(rvs[1:] == rvs_ff[:-1]))

    def test_sobol_engine_vs_scipy_qmc_sobol_lds(self):
        d = 10
        n = 128

        sobol_engine = SobolEngine(d=d, scramble=False, sequence=SequenceNum.NEW_JOE_KUO)
        scipy_qmc_sobol_engine = Sobol(d=d, scramble=False)

        sobol_engine_rvs = sobol_engine.random(n)
        scipy_qmc_sobol_engine_rvs = scipy_qmc_sobol_engine.random(n)

        np.testing.assert_array_equal(sobol_engine_rvs, scipy_qmc_sobol_engine_rvs)

    def test_sobol_engine_vs_scipy_qmc_sobol_lds_scrambled(self):
        d = 10
        n = 128
        seed = 123

        np.random.seed(seed)
        sobol_engine = SobolEngine(d=d, scramble=True, sequence=SequenceNum.NEW_JOE_KUO, seed=seed)
        sobol_engine_rvs = sobol_engine.random(n)

        np.random.seed(seed)
        scipy_qmc_sobol_engine = Sobol(d=d, scramble=True, seed=seed)
        scipy_qmc_sobol_engine_rvs = scipy_qmc_sobol_engine.random(n)

        np.testing.assert_array_equal(sobol_engine_rvs, scipy_qmc_sobol_engine_rvs)

    def test_sobol_engine_reset(self):
        d = 10
        n = 128

        sobol_engine = SobolEngine(d=d, sequence=SequenceNum.NEW_JOE_KUO)
        sobol_engine_rvs = sobol_engine.random(n)

        sobol_engine.reset()
        sobol_engine_rvs_after_reset = sobol_engine.random(n)

        np.testing.assert_array_equal(sobol_engine_rvs, sobol_engine_rvs_after_reset)
