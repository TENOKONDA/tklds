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

import numpy.testing as npt
from tklds.constant import SequenceNum
from tklds.integrals.non_decomposable import _generate_points

NUM_POINTS = 1000
NUM_DIM = 50


def get_actual_expected_outputs(sequence: SequenceNum, test_function):
    """

    Parameters
    ----------
    sequence : SequenceNum
        sequence to be generated and compared
    test_function : callable
        function being called

    Returns
    -------
    expected, actual

    expected: float
        expected analytic integral
    actual: float
        actual numeric integral
    """
    u = _generate_points(sequence, num_points=NUM_POINTS, num_dim=NUM_DIM, num_skip_points=1)
    expected, actual = test_function(u)
    return expected, actual


def compare_outputs(sequence: SequenceNum, test_function, places: int):
    """
    Compare an array to an expected array value

    Parameters
    ----------
    sequence: SequenceNum
        sequence to be generated and compared
    test_function : callable
        function being called
    places : int
        number of places to compare to
    """
    expected, actual = get_actual_expected_outputs(sequence, test_function)
    npt.assert_almost_equal(expected, actual, decimal=places)
