# Spectral attribution stability in radar gesture recognition (COSC 750, Stages 3 and 4)

Code for the baselines and the attribution-stability pipeline (Stage 3) and the extended experiments (Stage 4) on Google Soli (`deep-soli`).
Author: Adedeji Oludayo, Bowie State University.

## What this repo does
1. **Data** – reads the 5,500 HDF5 sequences once (straight from the zip), checks them, and caches a `[N,4,32,32,32]` array (`data_report.json` records label consistency, value range, frames per file, files per session).
2. **Splits** – user-grouped 4-fold protocol built from the README's session arrangement (all sessions of a user stay together; validation users come from the training pool; asserts no user/session overlap). The background class (gesture 11, no user) is excluded.
3. **Baselines** – B3 majority, B1a PCA + regularised Gaussian Bayes, B1b RBF-SVM on pooled spectral features, B2 CNN-RNN (Wang et al. 2016 style, re-implemented), plus the project CNN. Same tuning budget (4 configs, chosen on validation users only), 4 folds x 5 repeats.
4. **Stability** – clean-correct cohort shared by two CNN seeds and the SVM; Grad-CAM and occlusion maps on the 8x8 Doppler x time grid; three frequency-domain perturbations matched by input SSIM; seed-to-seed floor; model-randomisation sanity checks; retention counts.

## Reproduce (one command)
```
pip install -r requirements.txt && pip install -e .
make reproduce DATA=/path/to/SoliData.zip DEVICE=cuda   # builds cache, runs baselines, writes results/table2.md
make stability DEVICE=cuda                               # perturbation / attribution experiment
make test                                                # unit tests (splits, leakage, perturbations, metrics, occlusion)
make smoke                                               # 40 s end-to-end check on synthetic data, no download
```
Outputs: `results/baselines_runs.csv` (one row per method x fold x repeat, with hyper-parameters, users, seeds), `results/table2.md|tex`, `results/paired_tests.csv`, `results/confusion.png`, `results/stability_runs.csv`, `results/stability_summary.csv`, `results/stability.png`.

## Things to check before trusting a result
* **Doppler axis.** The deep-soli README does not say which axis of the 32x32 map is Doppler. Run `python scripts/inspect_axes.py --cache cache` and set `DOPPLER_AXIS` in `src/soli_attr/config.py`. Band masking and smoothing depend on it.
* **Session -> user mapping** comes from the README (`config.USER_SESSIONS`). If your copy differs, edit it; `tests/test_core.py` checks every session lands in exactly one fold.
* **Label consistency.** `data_report.json` must show `files_with_nonconstant_label = 0`; otherwise sequence labels need a rule.
* The "leaky" random-split SVM row is a reference only; it is never used for comparisons.

## Protocol notes
Normalisers are fitted on training data only; perturbations are applied after the split and calibrated on validation users; test users are untouched until final scoring. Seeds are `100*fold + repeat`. Paired t-tests use the four fold means (n=4) with a Bonferroni-style threshold of 0.0125; with so few folds the p-values are indicative, not strong.

## Layout
`src/soli_attr/{config,data,splits,features,perturb,models,explain,stability}.py`, `scripts/`, `tests/`.

## Changes in the v2 stability run (Oct 29)
- `band_mask` is now a graded attenuation (width 2-16 bins, gain 1 -> 0); `floor_shift` clips at a fraction of each map's peak (the quantile version was a no-op).
- Severity is matched two ways: target input SSIM (0.99 ... 0.65) and target CNN accuracy drop (1, 2, 5, 10 points). The summary has a `calib` column and a `reached` column (whether calibration hit the target).
- SVM has its own seed-style floor (SVM vs SVM refit on 80% of the training pool).
- `paired_tests.csv` is computed over the 4 folds (n = 4); `table2_per_fold.csv` lists per-fold accuracy.


## Notes added for the final Stage 3 paper

- **Resampling.** Every sequence is linearly resampled along time to 32 frames (`resample_time` in `src/soli_attr/data.py`). This is needed for a fixed model input, a shared 8x8 explanation grid and fixed perturbation bins. It is lossy: gesture duration is discarded and downsampling slightly smooths fast Doppler changes.
- **RQ3 correlation.** `make rq3` (or `python scripts/rq3_correlation.py --results results`) writes `results/rq3_correlation.csv`: Spearman correlation between map similarity and CNN accuracy drop over all perturbed conditions and over fold-level means. Both quantities move with severity, so it shows maps degrade with accuracy, not that they warn of it earlier.
- **Limits.** Perturbations are simulated on the pooled Doppler-time map, not the raw signal chain (see the paper's limitations section).

## Stage 4 additions

`make stage4` (after `make reproduce`) runs `scripts/run_stage4.py` and `scripts/report_stage4.py`:

- Integrated Gradients (own implementation) as a second CNN attribution family, and a continuous SVM class score built from pairwise margins (the one-vs-rest score includes integer vote counts).
- Deletion faithfulness: accuracy after deleting the k top-ranked, k random or k bottom-ranked grid cells.
- Two radar-motivated impairments next to the Stage 3 perturbations: additive noise at a set SNR and static zero-Doppler clutter (both simulated).
- Sensitivity: attribution grid size (4, 8, 16), training-set size (25, 50, 100 percent) and a 30-configuration SVM search.
- Stage 4 uses 3 repeats per fold (seed = 100 x fold + repeat, as before). Outputs: `results/stage4_*.csv`.

Main Stage 4 results (12 runs = 4 folds x 3 repeats; details and caveats in `reports/COSC750_F26_Oludayo_Stage4_paper.pdf`):

- Hypothesis H (map similarity below the seed floor while accuracy has dropped by less than 2 points) is met in 0 of 1,800 CNN cases (Grad-CAM, occlusion, Integrated Gradients; five perturbations; ten severities). The SVM meets it in 31 of 600 cases, but its floor is 0.987.
- Deleting the top-ranked cells hurts accuracy far more than deleting random cells for every map (faithfulness).
- Map similarity and accuracy drop correlate at Spearman -0.87 to -0.89 across conditions, and -0.26 to -0.44 within one severity level.

Output files (all in `results/`):

| File | Content |
|---|---|
| `stage4_stability_runs.csv` | one row per run x perturbation x level x map (similarity, retention, accuracy) |
| `stage4_faithfulness.csv` | deletion test |
| `stage4_perclass.csv`, `stage4_grid.csv` | per-gesture stability; grid-size sensitivity |
| `stage4_curve.csv`, `stage4_svmbudget.csv` | learning curves; 30-configuration SVM search |
| `stage4_tables/*.tex`, `figs/*.pdf`, `stage4_numbers.json` | every table and figure of the paper's Stage 4 sections, written by `scripts/report_stage4.py` (including `figs/svm_stability.pdf`) |

The extended run took about 90 minutes on one Colab GPU. `python scripts/report_stage4.py --res results` rebuilds the tables and figures from the shipped CSVs in under a minute, without a GPU. `scripts/plot_data_examples.py` draws the data figure (frame montage, channels, Doppler-time map) from the cache. The unit tests (19) run with `PYTHONPATH=src pytest tests`.
