# Citing tklds

Use a **version-specific software citation** for the package and cite the **2026 research paper** when the direction-number construction, pathwise dependence diagnostic, or reported benchmarks are material to the work.

## Software citation

> Andres Oliva Denis, James Wheeldon, Ilya Manyakin, and Adrien Papaioannou. *tklds: High-Dimensional Sobol’ Sequences for Python*, version 0.2.0. Tenokonda UK, 2026. https://github.com/TENOKONDA/tklds

The software citation should state the exact version used. After Zenodo integration is enabled, add the version DOI to this record and retain the Zenodo concept DOI for version-independent references.

## Preferred research citation

> Andres Oliva Denis, James Wheeldon, Ilya Manyakin, and Adrien Papaioannou. *Sobol Direction Numbers for Quasi-Monte Carlo Simulation: Construction, Pathwise Dependence Diagnostics, and Financial Pricing Benchmarks*. Tenokonda UK, 30 April 2026. SSRN abstract 7040539. https://ssrn.com/abstract=7040539

## BibTeX

Machine-readable BibTeX entries are supplied in [`CITATION.bib`](CITATION.bib). GitHub also reads [`CITATION.cff`](CITATION.cff) and exposes a **Cite this repository** control when the file is present at the repository root.

## Citation policy

- Cite the software version used to produce results.
- Cite the 2026 paper for claims about `tkrg-a-ap5`, Property A, Property A′ on adjacent five-dimensional blocks, construction over GF(2), the spurious-variance diagnostic, or the published pricing and risk benchmarks.
- Do not attribute results from older Broda or Joe–Kuo comparison studies directly to `tklds` unless the relevant tklds implementation was actually used.
- Do not describe benchmark findings as universal dominance across all QMC applications.

## DOI status

No Zenodo software DOI is recorded at the time this file was prepared. The repository includes `.zenodo.json` so that metadata is ready when the GitHub–Zenodo integration is enabled. Once Zenodo archives a release:

1. add the concept DOI and release DOI to `CITATION.cff` as appropriate;
2. add the release DOI to the software BibTeX record;
3. add DOI badges to `README.md`; and
4. update the release notes and documentation citation page.
