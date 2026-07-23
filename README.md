# tklds — High-Dimensional Sobol’ Sequences for Python

[![PyPI](https://img.shields.io/pypi/v/tklds.svg)](https://pypi.org/project/tklds/)
[![Python](https://img.shields.io/pypi/pyversions/tklds.svg)](https://pypi.org/project/tklds/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://github.com/TENOKONDA/tklds/blob/main/LICENSE)

**tklds** (Tenokonda Low Discrepancy Sequences) is a Python package and direction-number dataset for generating, scrambling, and evaluating high-dimensional Sobol’ sequences for quasi-Monte Carlo (QMC) simulation.

The package includes the **tkrg-a-ap5** direction-number set for **50,000 dimensions**. The construction satisfies **Property A in every supported dimension** and **Property A′ on every block of five adjacent dimensions**. It also includes the `new-joe-kuo-6.21201` direction numbers for reference and comparison.

The library provides:

- batch and iterative Sobol’ point generation;
- a SciPy-compatible `SobolEngine` with optional scrambling;
- the `tkrg-a-ap5` and `new-joe-kuo-6.21201` direction-number datasets;
- diagnostics for high-dimensional sequence behaviour;
- worked notebooks for integration, stochastic processes, pricing, sensitivities, and risk; and
- reproducibility material for the accompanying 2026 paper.

> **Scope note:** Direction-number performance depends on the integrand, coordinate ordering, path construction, scrambling method, and sample size. The supplied research benchmarks identify material advantages for the tested high-dimensional path-simulation regimes; they do not establish that one direction-number set dominates every QMC problem.

## Install

### From PyPI

```bash
python -m pip install tklds
```

### From a source checkout

```bash
git clone https://github.com/TENOKONDA/tklds.git
cd tklds
python -m pip install .
```

### Editable development installation

```bash
git clone https://github.com/TENOKONDA/tklds.git
cd tklds
python -m pip install -e ".[examples,test]"
```

`tklds` requires Python 3.10 or later. The `examples` extra installs the packages used by the notebooks; the `test` extra installs the proposed test runner and coverage tooling.

## Quick start

### Generate a batch of Sobol’ points

```python
from tklds.constant import SequenceNum
from tklds.interface.generators import generate_lds_rvs

points = generate_lds_rvs(
    sequence=SequenceNum.TKRG_A_AP5,
    n=1_024,
    d=32,
    skip=0,
)

print(points.shape)  # (1024, 32)
```

### Use the SciPy-compatible engine

```python
from tklds.constant import SequenceNum
from tklds.interface.generators import create_sobol_lds_engine

engine = create_sobol_lds_engine(
    sequence=SequenceNum.TKRG_A_AP5,
    d=256,
    scramble=True,
    seed=1234,
)

# 2**10 points. Power-of-two sample sizes preserve the natural
# balance properties of a Sobol’ digital net.
points = engine.random_base2(m=10)
print(points.shape)  # (1024, 256)
```

### Generate points iteratively

```python
from tklds.constant import SequenceNum
from tklds.interface.generators import create_iterative_lds_generator

sequence = create_iterative_lds_generator(
    sequence=SequenceNum.TKRG_A_AP5,
    d=8,
)

first_batch = sequence.rvs(size=(16, 8))
second_batch = sequence.rvs(size=(16, 8))
```

## Available direction-number sets

| Identifier | Direction-number set | Supported dimensions | Intended use |
|---|---|---:|---|
| `SequenceNum.TKRG_A_AP5` | `tkrg-a-ap5` | 50,000 | Default high-dimensional Sobol’ construction supplied by tklds |
| `SequenceNum.NEW_JOE_KUO` | `new-joe-kuo-6.21201` | 21,201 | Established reference construction and compatibility comparisons |

The raw direction-number files are distributed with the package under [`tklds/direction_numbers`](https://github.com/TENOKONDA/tklds/tree/main/tklds/direction_numbers).

## Choosing a generation method

Use `generate_lds_rvs` for a direct array of points, `create_iterative_lds_generator` when points are consumed in successive batches, and `create_sobol_lds_engine` when interoperability with the SciPy QMC interface is useful.

For deterministic Sobol’ experiments, use `scramble=False`. For replicate-based error estimation, use independent scrambled engines with controlled seeds. Where the balance properties of the underlying digital net matter, prefer `n = 2**m` and `random_base2(m)`.

## Examples and reproducibility

The repository separates introductory examples from paper-specific reproduction material:

- [`notebooks/01_quickstart.ipynb`](https://github.com/TENOKONDA/tklds/blob/main/notebooks/01_quickstart.ipynb) — package introduction;
- [`notebooks/02_integral.ipynb`](https://github.com/TENOKONDA/tklds/blob/main/notebooks/02_integral.ipynb) — numerical integration;
- [`notebooks/05_spurious_variance.ipynb`](https://github.com/TENOKONDA/tklds/blob/main/notebooks/05_spurious_variance.ipynb) — transformed-coordinate dependence diagnostic;
- [`notebooks/06_brownian_motion.ipynb`](https://github.com/TENOKONDA/tklds/blob/main/notebooks/06_brownian_motion.ipynb) — process simulation;
- [`notebooks/07_sobol_engine_examples.ipynb`](https://github.com/TENOKONDA/tklds/blob/main/notebooks/07_sobol_engine_examples.ipynb) — SciPy-compatible engine usage; and
- [`notebooks/paper_notebooks`](https://github.com/TENOKONDA/tklds/tree/main/notebooks/paper_notebooks) — figures and benchmarks associated with the 2026 paper.

Notebook dependencies are installed with:

```bash
python -m pip install -e ".[examples]"
```

## Research basis

The current technical reference is:

> Andres Oliva Denis, James Wheeldon, Ilya Manyakin, and Adrien Papaioannou. *Sobol Direction Numbers for Quasi-Monte Carlo Simulation: Construction, Pathwise Dependence Diagnostics, and Financial Pricing Benchmarks*. Tenokonda UK, 30 April 2026. SSRN abstract 7040539.

The paper introduces `tkrg-a-ap5`, explains its construction over GF(2), develops a pathwise dependence diagnostic, and reports deterministic pricing and risk benchmarks under standard chronological discretisation.

- [Read the paper on SSRN](https://ssrn.com/abstract=7040539)
- [Citation instructions](https://github.com/TENOKONDA/tklds/blob/main/CITATION.md)
- [Machine-readable citation](https://github.com/TENOKONDA/tklds/blob/main/CITATION.cff)
- [BibTeX entries](https://github.com/TENOKONDA/tklds/blob/main/CITATION.bib)

## Testing

The repository contains unit tests for the public generator interfaces, iterative generation, the SciPy-compatible Sobol engine, numerical integration utilities, stochastic-process components, and the spurious-variance diagnostic.

Run the current suite from a source checkout with:

```bash
python -m unittest discover -s tklds/tests -p "test_*.py"
```

Automated pull-request and release-gating workflows are a high-priority follow-on item. Until those workflows are added, maintainers should run the complete suite manually before tagging a release.

## Project links

- [Source code](https://github.com/TENOKONDA/tklds)
- [PyPI package](https://pypi.org/project/tklds/)
- [Issue tracker](https://github.com/TENOKONDA/tklds/issues)
- [Release history](https://github.com/TENOKONDA/tklds/releases)
- [Tenokonda](https://www.tenokonda.com/)

## Licence

`tklds` is distributed under the [BSD 3-Clause License](https://github.com/TENOKONDA/tklds/blob/main/LICENSE).
