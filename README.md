# Spectral attribution stability in radar gesture recognition (COSC 750, Stage 3)

Code for the Stage 3 baselines and the attribution-stability pipeline on Google Soli (`deep-soli`).
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
- **Limits and Stage 4.** Perturbations are simulated, not real radar impairments. Planned: physically grounded impairments and natural cross-user shift, Integrated Gradients, a margin-based SVM score, a faithfulness check against Doppler bins with gesture energy, equal SVM tuning, and a protocol frozen before running.
