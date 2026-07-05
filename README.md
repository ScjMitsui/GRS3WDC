# GRS3WDC

Reference implementation for the experiments in **"Three-way decision with
granular rough sets"** (Luo, Hu, Shi, Yao; *Applied Soft Computing* 180, 2025,
[doi:10.1016/j.asoc.2025.113344](https://doi.org/10.1016/j.asoc.2025.113344)).

Implements the **Granular Rough Sets Three-Way Decision Classifier (GRS3WDC)**
in its two instantiations from the paper: neighborhood-induced (`GRS3WDC-N`,
Section 3.3) and maximal-clique-induced (`GRS3WDC-MC`, Section 3.4) granular
spaces, benchmarked against SVM/Decision Tree/KNN/Naive Bayes.

## Layout

| Path | Purpose | Paper reference |
|---|---|---|
| `src/grs3wdc/granules/neighborhood.py` | GRS3WDC-N | Section 3.3, Definitions 12-15 |
| `src/grs3wdc/granules/maximal_clique.py` | GRS3WDC-MC | Section 3.4, Definitions 17-20 |
| `src/grs3wdc/baselines.py` | SVM / Decision Tree / KNN / Naive Bayes | Section 5.2 |
| `src/grs3wdc/data.py` | Dataset loading for the six benchmarks | Table 8 |
| `src/grs3wdc/evaluation.py` | Repeated holdout + three-way metric aggregation | Section 5.2 |
| `configs/experiments.yaml` | The exact `(beta, alpha)` grids reported | Section 5.2, Tables 9-10 |
| `experiments/run_experiment.py` | Run one dataset/model/config | -- |
| `experiments/hyperparameter_search.py` | Grid search `(beta, alpha)` | -- |
| `experiments/run_all.py` | Reproduce Tables 9-10 end to end (Docker entrypoint) | Tables 9-10 |
| `experiments/generate_figures.py` | Per-dataset metric bar charts | Fig. 9 |
| `data/*.csv` | Sonar, Breast Cancer, Banknote, Image, Phishing, Letter | Table 8 |
| `archive/` | Everything not needed to reproduce the final results (see below) | -- |

## Quickstart

```bash
pip install -e .

# Single run
python -m experiments.run_experiment --dataset banknote --model grs3wdc-n --beta 1 --alpha 0.82
python -m experiments.run_experiment --dataset breast --model svm

# Hyperparameter search
python -m experiments.hyperparameter_search --model grs3wdc-n --datasets banknote sonar --beta-values 0.5 1 2 3 --alpha-values 0.80 0.82 0.85

# Reproduce Table 9/10 for a subset of datasets (full run over all 6 can be slow, see below)
python -m experiments.run_all --datasets banknote sonar --n-runs 5

# Render Fig. 9-style bar charts from the results produced above
python -m experiments.generate_figures --datasets banknote sonar
```

## Implementation note

`neighborhood.py` predicts using a matched granule's **center's own label**,
while `maximal_clique.py` predicts using the granule's **aggregated majority
label**. This asymmetry exists in the original `twdnp.py`/`twdmp.py` scripts
that produced the paper's results, so it has been preserved rather than
"fixed" during consolidation -- changing it would change the reported numbers.
