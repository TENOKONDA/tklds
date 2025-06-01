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

import traceback
import unittest

import numpy as np
from tklds.generators.iterative_lds import _load_iterative_lds_tkrg_a_ap5
from tklds.processes.brownian_motion import _multivariate_brownian_motion, _normal_increments

NUM_DIM = 5
TKRG_A_AP5_ITERATIVE_LDS = _load_iterative_lds_tkrg_a_ap5(5, max_points=2 ** 32)
INITIAL_VALUE = np.zeros((1, NUM_DIM))
MU = np.zeros((1, NUM_DIM))
COV = np.array([[1, 0, 0, 0, 0],
                [0, 2, 0, 0, 0],
                [0, 0, 3, 0, 0],
                [0, 0, 0, 4, 0],
                [0, 0, 0, 0, 5]])
NUM_TIMESTEPS = 200
U = TKRG_A_AP5_ITERATIVE_LDS.rvs(NUM_TIMESTEPS)


class TestBrownianMotion(unittest.TestCase):

    def test_multivariate_normal_increments(self):

        np.random.seed(9123)
        a_u = np.random.uniform(0, 1, (10000, 3))

        a_mu = np.array([[1, 2, 3]])
        a_cov = np.array([
            [1, 0, 0],
            [0, 3, 1],
            [0, 1, 5]])

        a_x = _normal_increments(a_mu=a_mu, a_cov=a_cov, a_u=a_u)
        a_actual_cov = np.cov(a_x.T, rowvar=True)

        a_abs_diff = a_actual_cov - a_cov
        self.assertLess(np.max(a_abs_diff), 0.064)

    def test_multivariate_brownian_motion_invalid_number_rows_mu(self):
        a_mu = np.zeros((2, 3))
        a_x0 = np.zeros((1, 3))

        a_cov = np.array([
            [2.0, 0.6, 1.4],
            [0.6, 1.5, 0.5],
            [1.4, 0.5, 3.3]
        ])

        nt = 100000
        np.random.seed(19123)
        a_u = np.random.uniform(0, 1, (nt, 3))
        with self.assertRaises(ValueError):
            try:
                _multivariate_brownian_motion(a_x0=a_x0, a_mu=a_mu, a_cov=a_cov, a_u=a_u)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Invalid number of rows of 'mu', expected 1, actual: {a_mu.shape[0]}."
        self.assertIn(expected_error, error_str)

    def test_multivariate_brownian_motion_invalid_cov_shape(self):
        a_mu = np.zeros((1, 3))
        a_x0 = np.zeros((1, 3))

        a_cov = np.array([
            [2.0, 0.6, 1.4],
            [0.6, 1.5, 0.5],
            [1.4, 0.5, 3.3],
            [2.3, 5.6, 3.4]
        ])

        nt = 100000
        np.random.seed(19123)
        a_u = np.random.uniform(0, 1, (nt, 3))
        with self.assertRaises(ValueError):
            try:
                _multivariate_brownian_motion(a_x0=a_x0, a_mu=a_mu, a_cov=a_cov, a_u=a_u)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        ndim = a_mu.shape[1]
        expected_error = f"Invalid 'cov' shape, expected: ({ndim},{ndim}), actual: {a_cov.shape}."
        self.assertIn(expected_error, error_str)

    def test_multivariate_brownian_motion_invalid_a_u_column_num(self):
        a_mu = np.zeros((1, 3))
        a_x0 = np.zeros((1, 3))

        a_cov = np.array([
            [1, 0, 0],
            [0, 3, 1],
            [0, 1, 5]])

        nt = 100000
        np.random.seed(19123)
        a_u = np.random.uniform(0, 1, (nt, 4))
        with self.assertRaises(ValueError):
            try:
                _multivariate_brownian_motion(a_x0=a_x0, a_mu=a_mu, a_cov=a_cov, a_u=a_u)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        ndim = a_mu.shape[1]
        expected_error = f"Invalid number of columns of 'u', expected {ndim}, actual: {a_u.shape[1]}."
        self.assertIn(expected_error, error_str)

    def test_multivariate_brownian_motion_invalid_a_u_low_value(self):
        a_mu = np.zeros((1, 3))
        a_x0 = np.zeros((1, 3))

        a_cov = np.array([
            [1, 0, 0],
            [0, 3, 1],
            [0, 1, 5]])

        nt = 100000
        np.random.seed(19123)
        a_u = np.random.uniform(0, 1, (nt, 3))
        a_u[0, 1] = -0.1
        with self.assertRaises(ValueError):
            try:
                _multivariate_brownian_motion(a_x0=a_x0, a_mu=a_mu, a_cov=a_cov, a_u=a_u)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        u_min, u_max = np.min(a_u), np.max(a_u)

        expected_error = f"Invalid uniform variates, expected values in range (0,1) [exclusive], actual: " \
                         f"{u_min}, {u_max}."
        self.assertIn(expected_error, error_str)

    def test_multivariate_brownian_motion_invalid_a_u_high_value(self):
        a_mu = np.zeros((1, 3))
        a_x0 = np.zeros((1, 3))

        a_cov = np.array([
            [1, 0, 0],
            [0, 3, 1],
            [0, 1, 5]])

        nt = 100000
        np.random.seed(19123)
        a_u = np.random.uniform(0, 1, (nt, 3))
        a_u[0, 1] = 1.1
        with self.assertRaises(ValueError):
            try:
                _multivariate_brownian_motion(a_x0=a_x0, a_mu=a_mu, a_cov=a_cov, a_u=a_u)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        u_min, u_max = np.min(a_u), np.max(a_u)

        expected_error = f"Invalid uniform variates, expected values in range (0,1) [exclusive], actual: " \
                         f"{u_min}, {u_max}."
        self.assertIn(expected_error, error_str)
