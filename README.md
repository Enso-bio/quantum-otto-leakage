# Energy-weighted leakage in a symmetry-preserving quantum Otto engine

Reproducibility repository for the theoretical study by **Enso O. Torres Alegre**.

This repository contains the Python code, numerical tables, and figure-generation scripts associated with the manuscript **“Energy-weighted leakage limits work extraction in a symmetry-preserving quantum Otto engine.”** No experimental data are used.

## Quick start

### Local Python

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python validate_results.py
python paper_consistency_check.py
python make_figures.py
python make_bias_figure.py
```

### Full recomputation

```bash
python run_study.py
python additional.py
python bias_tls_compare.py
python verify_independent.py
python validate_results.py
python paper_consistency_check.py
python make_figures.py
python make_bias_figure.py
```

The full recomputation is substantially slower than the standard audit. The consistency checker uses the shipped tables for most claims and reruns only one targeted $\tau=10$ propagation.

### Google Colab

Open `COLAB_REPRODUCE.ipynb` and run all cells. The notebook clones this public repository directly from:

`https://github.com/Enso-bio/quantum-otto-leakage`

By default it performs a fast audit. Set `FULL_RECOMPUTE = True` to regenerate the numerical study from the Hamiltonian.

## Repository structure

- `engine.py` — spatial Hamiltonian, thermodynamic bookkeeping, and core numerical routines.
- `run_study.py` — reference spectrum and finite-time protocol sweeps.
- `additional.py` — temperature maps, equal-peak-speed comparison, and finite-contact calculations.
- `bias_tls_compare.py` — full-spatial versus biased two-level comparison and loss decomposition.
- `verify_independent.py` — independent full-grid Crank–Nicolson checks and algebraic random-unitary verification.
- `validate_results.py` — aggregate thermodynamic and numerical validation.
- `paper_consistency_check.py` — checks the numerical values quoted in the manuscript against the shipped tables and reruns one targeted $\tau=10$ propagation for the immediate-channel decomposition.
- `make_figures.py`, `make_bias_figure.py` — regenerate the manuscript figures.
- `data/` — raw CSV/JSON numerical outputs used for figures and quoted results.
- `figures/` — generated PDF/PNG figures for comparison with regenerated outputs.
- `COLAB_REPRODUCE.ipynb` — Colab workflow.
- `requirements.txt` — Python dependencies.

## Reproducibility notes

The reference calculations use deterministic numerical integration and diagonalization. The manuscript reports grid/basis convergence and an independent full-grid propagation check. Run `validate_results.py` and `paper_consistency_check.py` before comparing regenerated figures with the manuscript.

## Citation

If you use this code or the accompanying numerical data, please cite the associated manuscript and reference this repository:

**Enso O. Torres Alegre.** *Energy-weighted leakage limits work extraction in a symmetry-preserving quantum Otto engine.* Code and numerical data: https://github.com/Enso-bio/quantum-otto-leakage

`CITATION.cff` contains machine-readable citation metadata for GitHub's **Cite this repository** function.

## License

This software is released under the MIT License. See `LICENSE`.
