# ------------------------------------------------------------------------------------------------------------------
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
# ------------------------------------------------------------------------------------------------------------------

"""

You can find here some examples of tools to incorporate into the documentation:


- Maths:

.. math::

  W^{3\\beta}_{\\delta_1 \\rho_1 \\sigma_2} \\approx U^{3\\beta}_{\\delta_1 \\rho_1}

- Images:

.. image:: ../../resources/imag/jupyter_logo.png
  :width: 100
  :align: center
  :alt: jupyter_logo


- Matplotlib plots:

.. plot::
    :include-source: True

    import numpy as np
    import matplotlib.pyplot as plt
    # evenly sampled time at 200ms intervals
    t = np.arange(0., 5., 0.2)
    # red dashes, blue squares and green triangles
    plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
    plt.show()


- Graph Visualization:

.. graphviz::

   digraph Flatland {
      a  [shape=ellipse,label="test",width=0.4,fixedsize=true,image="../../resources/imag/jupyter_logo.png"]
      b  [shape=polygon,sides=5]
      c  [shape=polygon,sides=6]
      a -> b -> c -> g;

      g [peripheries=3,color=yellow];
      s [shape=invtriangle,peripheries=1,color=aquamarine3,style=filled];
      w [shape=triangle,peripheries=1,color=midnightblue];

      }


- Gallery:

.. include::
    auto_examples/index.rst

- :doc:`Link to gallery <auto_examples/index>`

"""
