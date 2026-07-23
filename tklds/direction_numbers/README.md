# Direction-number datasets

This directory contains the direction-number data distributed with `tklds`.

## `tkrg-a-ap5`

`tkrg-a-ap5` supports 50,000 dimensions and is the default Sobol’ direction-number construction used by the package. It satisfies Property A in every supported dimension and Property A′ on every block of five adjacent dimensions.

Files:

- `tkrgsobol_a_ap5_50000` — text representation;
- `tkrgsobol_a_ap5_50000.pickle` — packaged binary representation used by the implementation.

## `new-joe-kuo-6.21201`

The Joe–Kuo direction-number set is included as an established reference construction and for compatibility comparisons.

Files:

- `new-joe-kuo-6.21201` — text representation;
- `new-joe-kuo-6.21201.pickle` — packaged binary representation used by the implementation.

## Technical reference

The current project paper is:

> Andres Oliva Denis, James Wheeldon, Ilya Manyakin, and Adrien Papaioannou. *Sobol Direction Numbers for Quasi-Monte Carlo Simulation: Construction, Pathwise Dependence Diagnostics, and Financial Pricing Benchmarks*. Tenokonda UK, 30 April 2026. SSRN abstract 7040539.

Paper: https://ssrn.com/abstract=7040539
Repository: https://github.com/TENOKONDA/tklds

The paper’s benchmark results apply to the tested deterministic high-dimensional path-simulation regimes. They should not be read as a universal ranking of direction-number sets for every QMC integrand or construction.
