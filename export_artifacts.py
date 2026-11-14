#!/usr/bin/env python
"""EXPORT (no Jupyter needed): builds everything infer.py needs from an existing `dr_work` folder.

  python export_artifacts.py --root C:/path/to/dr_work            # writes ./weights and ./sample_data next to this file

Needs inside dr_work:  cache/manifest.csv, cache/master512/*.png, save/{locked_model.pt,locked_model.json,ref_cdf.npy}, save/collect/train/chunk_*.pkl
It rebuilds the same sampling plan as the notebook (same seeds), checks it against the collected chunks, refits the policy models
(identical recipe to the final test) and saves policy.joblib. Training is NOT repeated.
"""
import argparse, os, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd, cv2, joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SEED, N_RAND, CM, CF, LAM, TOL = 0, 4, 5.0, 1.0, 0.5, 0.05       # same constants as the notebook


def read512(p): return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def build_plan(imgs, dr_rl, idx, n, tag, seed):                  # identical to the notebook's build_plan
    rng = np.random.default_rng(seed); rows = []
    for k in range(n):
        r = imgs.loc[idx[int(rng.integers(len(idx)))]]; spec = dr_rl.sample_spec(rng)
        rows.append(dict(sid=f'{tag}{k}', id=r.id, label=int(r.label), p_clean=float(r.p_clean), path=r.path512, spec=spec,
                         seed=int(rng.integers(2 ** 31)), rand_acts=dr_rl.random_actions(rng, N_RAND), families=sorted(dr_rl.families(spec)),
                         kind='clean' if not spec else 'degraded', split=tag))
    return rows


def load_chunks(folder, n_expected):
    import pickle
    fs = sorted(Path(folder).glob('chunk_*.pkl')); assert fs, f'no chunk files in {folder}'
    D = [pickle.load(open(f, 'rb')) for f in fs]
    out = {k: np.concatenate([d[k] for d in D]) for k in ['state', 'p0', 'p', 'act']}
    assert len(out['p0']) == n_expected, f'{len(out["p0"])} samples collected, expected {n_expected}'
    return out


def rows_fn(D, meta, idx, thr):                                  # candidate 0 = do nothing, 1..5 library, 6..9 random actions
    n = len(idx); acts = np.concatenate([np.zeros((n, 1, 4)), D['act'][idx]], axis=1)
    pn = np.concatenate([D['p0'][idx][:, None], D['p'][idx]], axis=1)
    y, pc = meta.label.values[idx][:, None], meta.p_clean.values[idx][:, None] >= thr
    cost = (CM * ((y == 1) & pc & (pn < thr)) + CF * ((y == 0) & ~pc & (pn >= thr))).astype(np.float32)
    st = np.c_[D['state'][idx], D['p0'][idx]]
    X = np.concatenate([np.repeat(st[:, None, :], 10, axis=1), acts], axis=2)
    return st, X, cost


def fit_ens(X, y, k=3):
    ms = []
    for s in range(k):
        r = np.random.default_rng(s).integers(0, len(X), len(X))
        ms.append(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40, random_state=s).fit(X[r], y[r]))
    return ms


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='./dr_work'); ap.add_argument('--out', default=str(HERE)); ap.add_argument('--device', default=None)
    ap.add_argument('--n-train', type=int, default=3000, help='must equal N_TRAIN used for collection in the notebook')
    a = ap.parse_args(argv)
    ROOT = Path(a.root).resolve(); CACHE, SAVE = ROOT / 'cache', ROOT / 'save'; OUT = Path(a.out)
    os.environ['DR_REF_PATH'] = str(SAVE / 'ref_cdf.npy')
    import dr_ops, dr_rl, infer
    from infer import DEMO_DEGRADATIONS

    # ---- data: policy split (images the classifier never trained on) + their 512 px masters
    m = pd.read_csv(CACHE / 'manifest.csv')
    imgs = m[(m.dataset == 'aptos') & (m.split == 'policy')].copy().reset_index(drop=True)
    assert len(imgs), "no 'policy' split in manifest.csv (was POLICY_HOLDOUT_FRAC = 0?)"
    M512 = CACHE / 'master512'; M512.mkdir(exist_ok=True)
    imgs['path512'] = [str(M512 / f'{i}.png') for i in imgs.id]
    for src, dst in zip(imgs.master_path, imgs.path512):
        if not os.path.exists(dst): cv2.imwrite(dst, cv2.resize(cv2.imread(src), (512, 512), interpolation=cv2.INTER_AREA))

    # ---- locked classifier -> clean probabilities
    clf = infer.Classifier(SAVE, a.device); thr = clf.thr
    imgs['p_clean'] = np.concatenate([clf([read512(p) for p in imgs.path512[i:i + 128]]) for i in range(0, len(imgs), 128)])
    pcn = imgs.p_clean >= thr
    print(f'{len(imgs)} images | clean sensitivity {pcn[imgs.label == 1].mean():.3f} specificity {(~pcn[imgs.label == 0]).mean():.3f}'
          '   (notebook printed: 1026 images | 0.857 | 0.956)')

    # ---- rebuild the training plan exactly as in the notebook and verify it against the collected chunks
    tr_i, _ = train_test_split(imgs.index, test_size=0.3, stratify=imgs.label, random_state=SEED)
    plan = build_plan(imgs, dr_rl, list(tr_i), a.n_train, 'tr', 1); meta = pd.DataFrame(plan)
    D = load_chunks(SAVE / 'collect' / 'train', len(plan))
    for k in sorted({0, 1, 2, len(plan) // 2, len(plan) - 1}):
        p = plan[k]; r = dr_rl.work((p['path'], p['spec'], p['seed'], []))
        ok_state = np.allclose(r['state'], D['state'][k], atol=1e-3, rtol=1e-3)
        ok_p0 = abs(float(clf([r['deg']])[0]) - float(D['p0'][k])) < 0.03
        if not (ok_state and ok_p0):
            raise SystemExit(f'CHECK FAILED at sample {k}: rebuilt plan does not match the collected data '
                             f'(state ok={ok_state}, p0 ok={ok_p0}). Use the same manifest.csv / dr_work as the notebook run.')
    print('plan check passed: rebuilt samples reproduce the collected states and classifier outputs')

    # ---- gate thresholds (rule policy) from clean calibration images
    cal = imgs.loc[tr_i].sample(min(300, len(tr_i)), random_state=0)
    F_cal = np.stack([dr_rl.state_features(read512(p)) for p in cal.path512])
    lo, hi = np.quantile(F_cal, 0.01, axis=0), np.quantile(F_cal, 0.99, axis=0)

    # ---- refit policy models on ALL training samples (same as fit_for(None))
    st, X, cost = rows_fn(D, meta, np.arange(len(meta)), thr)
    surrogate = fit_ens(X.reshape(-1, X.shape[-1]), cost.reshape(-1))
    selector = [HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=30,
                                              random_state=c).fit(st, cost[:, c]) for c in range(6)]
    W = OUT / 'weights'; W.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(lo=lo, hi=hi, surrogate=surrogate, selector=selector, TOL=TOL, LAM=LAM,
                     feature_names=dr_rl.FEATURE_NAMES + ['p0'], threshold=float(thr)), W / 'policy.joblib')
    for f in ('locked_model.pt', 'locked_model.json', 'ref_cdf.npy'): shutil.copy(SAVE / f, W / f)

    # ---- demo images: 2 clean + degradations that cause a new miss on a correctly classified referable image
    S = OUT / 'sample_data'; S.mkdir(parents=True, exist_ok=True); JQ = [cv2.IMWRITE_JPEG_QUALITY, 95]
    for lbl, tag in ((1, 'referable'), (0, 'healthy')):
        sel = imgs[(imgs.label == lbl) & ((imgs.p_clean >= thr) == bool(lbl))]
        if len(sel): cv2.imwrite(str(S / f'clean_{tag}_{sel.iloc[0].id}.jpg'), cv2.imread(sel.iloc[0].path512), JQ)
    cands = imgs[(imgs.label == 1) & (imgs.p_clean >= thr)]; cands = cands.sample(min(60, len(cands)), random_state=0)
    for dname, spec in DEMO_DEGRADATIONS.items():
        for _, r in cands.iterrows():
            x = read512(r.path512); g = np.random.default_rng(0)
            for n_, kw in spec: x = dr_ops.DEG[n_](x, g, **kw)
            x = dr_rl.to512(x)
            if clf([x])[0] < thr:
                cv2.imwrite(str(S / f'{dname}_{r.id}.jpg'), cv2.cvtColor(x, cv2.COLOR_RGB2BGR), JQ); break
    print('weights/:', sorted(p.name for p in W.iterdir())); print('sample_data/:', sorted(p.name for p in S.iterdir()))


if __name__ == '__main__':
    main()
