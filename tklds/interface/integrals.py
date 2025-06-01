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

from inspect import getmembers, isfunction
from typing import Callable, List, Tuple
from tklds.constant import SequenceNum, EffectiveDimensionNum
from tklds.integrals.effective_dimensions import get_dimension_type
from tklds.integrals import additive, multiplicative, non_decomposable


def get_sequence_enum() -> SequenceNum:
    """
    Get the enum of valid low discrepancy direction number sequence

    Returns
    -------
    SequenceNum
        Enum of valid sequences
    """
    return SequenceNum


def get_effective_dimension_enum() -> EffectiveDimensionNum:
    """
    Get the list of valid low discrepancy direction number sequence names

    Returns
    -------
    EffectiveDimensionNum
        Enum of the valid sequences
    """
    return EffectiveDimensionNum


def get_integral_by_name(integral_name: str) -> List[Callable]:
    """
    Get an integral test function by name

    Parameters
    ----------
    integral_name : str
        the name of the integral test functions from the tklds.integrals module

    Returns
    -------
    List[Callable]
    """
    if integral_name in get_additive_integral_lst():
        module = additive
    elif integral_name in get_multiplicative_integrals_lst():
        module = multiplicative
    elif integral_name in get_non_decomposable_integrals_lst():
        module = non_decomposable
    else:
        raise ValueError(f"Integral with {integral_name=} not found!")

    integral_function_lst = [obj[1] for obj in getmembers(module, isfunction) if obj[0] == integral_name][0]
    return integral_function_lst


def get_integrals_name_and_type_lst() -> List[Tuple[str, str]]:
    """
    Get the list of test integrals and their types (one of: additive, multiplicative, non_decomposable) in the format:
    (integral_name, integral_type).

    Returns
    -------
    List[Tuple[str,str]]
        list of (integral_name, integral_type) tuples
    """
    additive_lst = [(name, "additive") for name in get_additive_integral_lst()]
    multiplicative_lst = [(name, "multiplicative") for name in get_multiplicative_integrals_lst()]
    non_decomposable_lst = [(name, "non_decomposable") for name in get_non_decomposable_integrals_lst()]

    return additive_lst + multiplicative_lst + non_decomposable_lst


def get_additive_integral_lst() -> List[str]:
    """
    Get names of test integrals whose integrand is of the 'additive' form.

    Returns
    -------
    List[str]
        list of additive integral names
    """
    return [obj[0] for obj in getmembers(additive, isfunction)]


def get_multiplicative_integrals_lst() -> List[str]:
    """
    Get names of test integrals whose integrand is of the 'multiplicative' form.

    Returns
    -------
    List[str]
        list of multiplicative integral names
    """
    return [obj[0] for obj in getmembers(multiplicative, isfunction)]


def get_non_decomposable_integrals_lst() -> List[str]:
    """
    Get names of test integrals whose integrand is of the 'non_decomposable' form.

    Returns
    -------
    List[str]
        list of non_decomposable integral names
    """
    return [obj[0] for obj in getmembers(non_decomposable, isfunction)]


def get_integral_effective_dimension_function(effective_dimension: EffectiveDimensionNum) -> Callable:
    """
    Get an integral effective dimension function by name

    Parameters
    ----------
    effective_dimension : EffectiveDimensionNum
        the integral effective dimension function

    Returns
    -------
    Callable
    """
    if not isinstance(effective_dimension, EffectiveDimensionNum):
        raise TypeError(f"Expected 'effective_dimension' to be an SequenceNum type, received: "
                        f"{type(effective_dimension).__name__}")
    return get_dimension_type(effective_dimension)
