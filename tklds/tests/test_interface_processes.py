import traceback
import unittest

import numpy as np
import scipy.stats as st
from tklds.constant import SequenceNum
from tklds.generators.iterative_lds import _load_iterative_lds_tkrg_a_ap5
from tklds.interface.generators import generate_lds_rvs
from tklds.interface.processes import multivariate_brownian_motion, spurious_variance


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


class TestIntefaceProcesses(unittest.TestCase):
    def test_multivariate_brownian_motion_wrong_type_initial_value(self):
        with self.assertRaises(TypeError) as e:
            try:
                multivariate_brownian_motion(a_x0="d", a_mu=MU, a_cov=COV, a_u=U)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Invalid type 'x0'.  Expected numpy.ndarray, actual: str"
        self.assertIn(expected_error, error_str)

    def test_multivariate_brownian_motion_wrong_type_mu(self):
        with self.assertRaises(TypeError) as e:
            try:
                multivariate_brownian_motion(a_x0=INITIAL_VALUE, a_mu="d", a_cov=COV, a_u=U)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Invalid type 'mu'.  Expected numpy.ndarray, actual: str"
        self.assertIn(expected_error, error_str)

    def test_multivariate_brownian_motion_wrong_type_mu(self):
        with self.assertRaises(TypeError) as e:
            try:
                multivariate_brownian_motion(a_x0=INITIAL_VALUE, a_mu="d", a_cov=COV, a_u=U)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Invalid type 'mu'.  Expected numpy.ndarray, actual: str"
        self.assertIn(expected_error, error_str)

    def test_multivariate_brownian_motion_wrong_type_cov(self):
        with self.assertRaises(TypeError) as e:
            try:
                multivariate_brownian_motion(a_x0=INITIAL_VALUE, a_mu=MU, a_cov="d", a_u=U)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Invalid type 'cov'.  Expected numpy.ndarray, actual: str"
        self.assertIn(expected_error, error_str)

    def test_multivariate_brownian_motion_wrong_type_u(self):
        with self.assertRaises(TypeError) as e:
            try:
                multivariate_brownian_motion(a_x0=INITIAL_VALUE, a_mu=MU, a_cov=COV, a_u="d")
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Invalid type 'u'. Expected numpy.ndarray, actual: str"
        self.assertIn(expected_error, error_str)

    def test_multivariate_brownian_motion_mismatching_dimensions_matrices(self):
        a_cov = np.array([[1, 2], [3, 4], [5, 6]])
        with self.assertRaises(ValueError):
            try:
                multivariate_brownian_motion(a_x0=INITIAL_VALUE, a_mu=MU, a_cov=a_cov, a_u=U)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = "Invalid 'cov' input. Expected square covariance matrix: 3 != 2"
        self.assertIn(expected_error, error_str)

    def test_multivariate_brownian_motion_cov_non_positive_definite(self):
        a_cov_non_positive_definite = np.array([[0.88826907, 0.42621251, 0.47455921, 0.70437439, 0.31480397],
                                                [0.42621251, 0.50744367, 0.21553875, 0.52174823, 0.28721659],
                                                [0.47455921, 0.21553875, 0.97502966, 0.46276767, 0.61544634],
                                                [0.70437439, 0.52174823, 0.46276767, 0.01142895, 0.31707795],
                                                [0.31480397, 0.28721659, 0.61544634, 0.31707795, 0.50330796]])

        with self.assertRaises(ValueError):
            try:
                multivariate_brownian_motion(a_x0=INITIAL_VALUE, a_mu=MU, a_cov=a_cov_non_positive_definite, a_u=U)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = "Invalid cov input.  Expected covariance matrix to be positive definite."
        self.assertIn(expected_error, error_str)

    def test_multivariate_brownian_motion_cov_non_symmetric(self):
        a_cov_non_symmetric = np.array([[5.08141801, 0.71243109, 0.52808072, 0.39573796, 0.52912011],
                                        [0.24, 5.28451037, 0.61157757, 0.64655309, 0.98345764],
                                        [0.52808072, 0.61157757, 5.71356248, 0.20341921, 0.96607259],
                                        [0.39573796, 0.64655309, 0.20341921, 5.71476688, 0.5832231],
                                        [0.52912011, 0.98345764, 0.96607259, 0.5832231, 5.46328537]])

        with self.assertRaises(ValueError):
            try:
                multivariate_brownian_motion(a_x0=INITIAL_VALUE, a_mu=MU, a_cov=a_cov_non_symmetric, a_u=U)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = "Invalid cov input. Expected symmetric covariance matrix."
        self.assertIn(expected_error, error_str)

    def test_multivariate_brownian_motion_cov_x0_shape_mismatch(self):
        a_cov_non_symmetric = np.array([[5.08141801, 0.71243109, 0.52808072, 0.39573796],
                                        [0.24, 5.28451037, 0.61157757, 0.64655309],
                                        [0.52808072, 0.61157757, 5.71356248, 0.20341921],
                                        [0.39573796, 0.64655309, 0.20341921, 5.71476688]])

        with self.assertRaises(ValueError):
            try:
                multivariate_brownian_motion(a_x0=INITIAL_VALUE, a_mu=MU, a_cov=a_cov_non_symmetric, a_u=U)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = "Dimension mismatch for parameters 'cov', 'x0', 'mu', 'u'."
        self.assertIn(expected_error, error_str)

    def test_multivariate_brownian_motion_2d(self):

        a_mu = np.zeros((1, 2))
        a_x0 = np.zeros((1, 2))
        a_cov = np.array([[1., 1.99999], [1.99999, 4.]])

        nt = 10000
        np.random.seed(0)
        a_u = st.uniform().rvs((nt, 2))
        a_x = multivariate_brownian_motion(a_x0=a_x0, a_mu=a_mu, a_cov=a_cov, a_u=a_u)

        a_increments = np.diff(a_x, axis=0)
        a_actual_cov = np.cov(a_increments.T)

        a_abs_diff = np.abs(a_actual_cov - a_cov)
        self.assertLess(np.max(a_abs_diff), 0.12)

    def test_multivariate_brownian_motion_3d(self):

        a_mu = np.zeros((1, 3))
        a_x0 = np.zeros((1, 3))

        a_cov = np.array([
            [2.0, 0.6, 1.4],
            [0.6, 1.5, 0.5],
            [1.4, 0.5, 3.3]
        ])

        nt = 100000
        np.random.seed(19123)
        a_u = np.random.uniform(0, 1, (nt, 3))
        a_x = multivariate_brownian_motion(a_x0=a_x0, a_mu=a_mu, a_cov=a_cov, a_u=a_u)

        a_increments = np.diff(a_x, axis=0)
        a_actual_cov = np.cov(a_increments.T)

        a_abs_diff = np.abs(a_actual_cov - a_cov)
        self.assertLess(np.max(a_abs_diff), 0.015)

    def test_spurious_variance(self):
        n = 1023
        n_skip = 1
        d_max = 1000
        a_u = generate_lds_rvs(sequence=SequenceNum.TKRG_A_AP5, n=n, d=d_max)[n_skip:]
        spurious_variance(a_u, verbose=False)

    def test_spurious_variance_type_error_u(self):
        n = 1023
        n_skip = 1
        d_max = 1000
        u_lst = list(generate_lds_rvs(sequence=SequenceNum.TKRG_A_AP5, n=n, d=d_max)[n_skip:])
        with self.assertRaises(TypeError):
            try:
                spurious_variance(u_lst, verbose=False)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Invalid type 'u'. Expected numpy.ndarray, actual: {type(u_lst).__name__}"
        self.assertIn(expected_error, error_str)
