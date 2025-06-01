Direction Numbers
==================

This folder contains the direction numbers used to initialize the Sobol sequences


tkrgsobol_a_ap5_50000
--------------------

This set of direction numbers has 50,000 dimensions, satisfying:
    * property A for all dimensions
    * property A' for 5 adjacent dimensions

For further details on their generation and the properties A and A' see (insert Arxiv link here).

new-joe-kuo-6.21201
--------------------

These contain 21,201 dimensions. Their generation is described in:

S. Joe and F. Y. Kuo, Constructing Sobol sequences with better two-dimensional projections, SIAM J. Sci. Comput. 30, 2635-2654 (2008).

Further details about these direction numbers and an alternative download source is available at:

https://web.maths.unsw.edu.au/~fkuo/sobol/index.html

[Direct download: https://web.maths.unsw.edu.au/~fkuo/sobol/new-joe-kuo-6.21201 ]

File format
===========

Both files follow the same format, the columns of the file:

* d - the index of the dimension
    - note that the smallest value is 2, as the first dimension is implicit.
* s - the **degree** of the primitive polynomial
* a - the **primitive polynomial coefficients** represented in terms of an integer value
* m_i - the **direction numbers** of the dimension

For a description of the terms and underlying theory of low discrepancy sequences consult the publication (insert Arxiv link here).