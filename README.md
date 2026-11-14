# Fix Before You Recognize: restoration policies for a locked diabetic-retinopathy classifier

AI61201 Visual Computing project. Authors: Disha Mukherjee (23CH10026), Soumyashree Nayak (23CH10099)

A frozen EfficientNet-B0 flags *referable DR* (grade >= 2) on fundus photos. Poor uploads (low light, haze, noise, ...) cause **new misses**. We learn to pick a restoration action per image (gamma / dehaze / NLM denoise, strength and order) **without retraining the classifier**.
Policies compared: `none`, `rule` (threshold checklist), `selector` (gradient boosting), `cma` (surrogate + CMA-ES, best).
Cost = 5 x new misses + 1 x new false referrals, both relative to the clean-image decision.

## Files
| File | Purpose |
|---|---|
| `infer.py` | **Inference / demo script** (run this) |
| `weights/` | `locked_model.pt/.json` (classifier + threshold 0.9101), `ref_cdf.npy` (clean colour reference), `policy.joblib` (selector, surrogate ensemble, rule gates) |
| `sample_data/` | demo fundus images: 2 clean, plus degraded ones that make the classifier miss (APTOS-2019 images, demo use only) |
| `dr_restoration_pipeline.ipynb` | **Training / experiments** notebook, outputs kept (full run: data, classifier, grids, collection, policies, final test) |
| `dr_common.py` | locked pipeline: crop/resize, preprocessing, EfficientNet-B0, threshold, metrics |
| `dr_ops.py` | 26 degradations, 13 restorers, picklable `SpecFn` |
| `dr_rl.py` | random degradation sampler, continuous-action restorer, 12 state features, collection worker |
| `requirements.txt` | dependencies |

(`dr_*.py` are written by the notebook's `%%writefile` cells; copies are included so inference runs without the notebook.)

## Run inference
```bash
pip install -r requirements.txt
python infer.py --input sample_data --policy cma --out sample_output/results.csv --save-restored sample_output/restored
python infer.py --input sample_data/<one clean_*.jpg file> --degrade lowlight --policy selector   # simulate a bad upload (use the real file name)
```
`--policy {cma,selector,rule,none}`, `--degrade {lowlight,haze,noise,underexposed,mixed}`. CPU is enough (~0.3 s/image rule/selector, ~0.7 s cma).
Output CSV: `p_clean` / `referable_clean` (only with `--degrade`: the decision on the undegraded image = the reference), `p_before`, `p_after` (probability of referable DR), decisions at the locked threshold, chosen action, time.
Input: any fundus photo (jpg/png); black border is cropped automatically.

## Reproduce training
1. `pip install -r requirements.txt`; set up Kaggle credentials (`kagglehub.login()`), datasets: `mariaherrerot/aptos2019`, `mariaherrerot/idrid-dataset`.
2. Run `dr_restoration_pipeline.ipynb` top to bottom (resumable; everything cached under `DR_ROOT`, default `./dr_work`):

| Sections | Step |
|---|---|
| 1-9 | datasets, master images, classifier (15 epochs, locked threshold) |
| 10-15 | degradation headroom, restoration grid, oracle gap |
| later | features, rule policy, compound/order tests, GB selector |
| data collection | 3000 train + 1000 test samples x 10 candidates |
| main/final test | surrogate + CMA-ES, final test |
| export | `python export_artifacts.py --root <dr_work>` -> `weights/policy.joblib`, `sample_data/` (no Jupyter needed; same as the notebook's last cell) |

Seeds fixed (`SEED = 0`). Splits: APTOS-train (65% classifier / 35% `policy` hold-out for policy development), APTOS-val (threshold), APTOS-test + IDRiD (external, final test only).

## Key results (final test, APTOS-test + IDRiD pooled, 3 random degradations per image; per-image cost, lower is better)
| Policy | Cost |
|---|---|
| do nothing | 0.391 |
| rule | 0.194 |
| GB selector | 0.237 |
| surrogate + CMA-ES | **0.174** |
| reference: selector with classifier feedback (upper bound, 5x classifier calls) | 0.125 |

Clean classifier: AUC 0.983 (APTOS-val), 0.984 (APTOS-test), 0.970 (IDRiD external).

Degraded samples, cost without -> with surrogate + CMA-ES: APTOS-test 0.385 -> 0.180; IDRiD 0.713 -> 0.276 (rescued 107 of 129 missed referable cases on IDRiD, 38 of 40 on APTOS-test).
Limitation: on already-clean images the policy is not free. On clean IDRiD, CMA-ES causes 5 new misses (new-miss rate 0.027) and the GB selector 9 (0.049); on clean APTOS-test CMA-ES causes no new misses (but 5 new false referrals; clean-group cost 0.014).

## Data and degradations
- Datasets (Kaggle): APTOS-2019 https://www.kaggle.com/datasets/mariaherrerot/aptos2019 and IDRiD https://www.kaggle.com/datasets/mariaherrerot/idrid-dataset (downloaded in notebook section 3 via `kagglehub`; not redistributed here except the few demo images).
- Demo degradations (`--degrade`, defined in `infer.py`): `lowlight` (dark=0.25), `haze` (t=0.6), `noise` (Poisson-Gaussian gain 1), `underexposed` (gamma 3), `mixed` (noise + vignette 0.9 + JPEG q40). Implementations: `dr_ops.py` (`@deg` functions).
- Evaluation degradations: random draws from 6 families (light, blur, noise, color, sharpen, compression; 1-3 stacked, 30% clean) defined in `dr_rl.py` (`COMP`, `sample_spec`); 3 random draws per test image, fixed seeds.
- Reading the demo: an image counts as *rescued* when `referable_clean` is True, `referable_before` is False (degradation caused a miss) and `referable_after` is True. The demo images are chosen to show such cases; the quantitative claim is the final-test table above (hundreds of images), not the handful of demo samples.
