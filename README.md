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

## Docker

```bash
docker build -t grs3wdc .
docker run --rm -v "$(pwd)/results:/app/results" grs3wdc --datasets banknote sonar --n-runs 5
```

The container's entrypoint is `experiments.run_all`; any of its CLI flags can
be passed after the image name. Mount `./results` to persist output.

**Note on runtime**: `GRS3WDC-MC` enumerates maximal cliques via NetworkX,
which does not scale well to large datasets -- the paper itself flags this as
a limitation. A full `run_all` over all six datasets (including Phishing and
Letter) at 20 runs can take a long time; use `--datasets`/`--n-runs` to scope
a demo run.

## Known data caveat

Table 8 reports 2310 instances for Image Segmentation; the local `image.csv`
has only 210 rows (it matches the small UCI "test" partition only, not the
combined train+test set used for the paper's reported numbers). Re-fetch the
full UCI dataset if you need to exactly reproduce that row of Tables 9-10.

## What's in `archive/`

The original project (copied from a Jupyter server) had ~15 notebooks and 6
near-duplicate model scripts accumulated over a year of experimentation.
Everything that didn't feed into the paper's final tables was moved here
as-is, for provenance:

- `archive/notebooks/` -- every original notebook, including the ones that
  produced Tables 9-10 (`hypertunning.ipynb`, `baseline.ipynb`,
  `validate_maximal.ipynb`, `validate_neighborhood.ipynb`,
  `visulization_Untitled.ipynb`) and ones that were abandoned dead ends
  (`initial.ipynb`, `test.ipynb`, `test_quan.ipynb`, `knntwdnp.ipynb`,
  `hypertunning-Copy1.ipynb` -- an abandoned weighted-voting granule scheme,
  `reduced_neighborhood.ipynb` -- an abandoned neighborhood/clique hybrid,
  `baseline-dl.ipynb` -- a Keras/keras-tuner FCNN baseline never reported in
  the paper).
- `archive/legacy_modules/` -- the 6 raw `.py` files (`twdneighborhood.py`,
  `twdmaximal.py`, `twdnp.py`, `twdmp.py`, `twdnpchev.py`, `twdmpchev.py`)
  that `src/grs3wdc/granules/*.py` now consolidates into two modules
  parameterized by `metric=` (euclidean/chebyshev) and `threshold_mode=`
  (percentile/absolute).
- `archive/data/` -- datasets that don't appear in Table 8 (`glass.csv`,
  `glass_o.csv`, `digits.csv`, `shuttle.csv`, `yeast.csv`, `breast_s.csv`,
  `sonar-Copy1.csv`) plus the original `visulization/` outputs.
- `archive/kt_tuner_dir/` -- Keras-tuner search cache from the abandoned DL
  baseline.

One dangling reference worth knowing about: `archive/notebooks/baseline.ipynb`
imports a `twdgranule` module that no longer exists anywhere in the project
history we copied. Based on its call signature (`run_model(X_train, y_train,
X_test, y_test, delta, alpha)`, an absolute delta rather than a percentile),
it was almost certainly an earlier name for what is now
`archive/legacy_modules/twdneighborhood.py`, generalized today as
`src/grs3wdc/granules/neighborhood.py` with `threshold_mode="absolute"`.

## Implementation note

`neighborhood.py` predicts using a matched granule's **center's own label**,
while `maximal_clique.py` predicts using the granule's **aggregated majority
label**. This asymmetry exists in the original `twdnp.py`/`twdmp.py` scripts
that produced the paper's results, so it has been preserved rather than
"fixed" during consolidation -- changing it would change the reported numbers.
