#!/usr/bin/env python
"""INFERENCE / DEMO SCRIPT  --  "Fix Before You Recognize" (diabetic-retinopathy case study)

For every fundus image:
  1. crop the black border, pad to square, resize to 512 px           (same as training-time masters)
  2. optional: apply a synthetic degradation (--degrade) to simulate a poor upload
  3. state = 12 image statistics + classifier probability p0 on the (possibly degraded) image
  4. policy picks a restoration action:  cma (surrogate + CMA-ES, best)  | selector (gradient boosting)
                                          | rule (threshold checklist)   | none (baseline)
  5. restore, re-classify with the LOCKED EfficientNet-B0, compare with the locked threshold

Usage (from this folder):
  python infer.py --input sample_data --policy cma --out sample_output/results.csv --save-restored sample_output/restored
  python infer.py --input my_image.jpg --degrade lowlight --policy selector

Needs ./weights/{locked_model.pt, locked_model.json, ref_cdf.npy, policy.joblib}.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, cv2, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

IMG_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
LIB_NAMES = ['none', 'gamma', 'dehaze', 'denoise', 'gamma+denoise', 'dehaze+denoise']
# action = [gamma strength, dehaze strength, denoise strength, order flag]; row 0 = do nothing (same as A_LIB in the notebook)
A_LIB = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.75, 0], [1, 0, 0.75, 0], [0, 1, 0.75, 0]], float)
FEATURES = ['noise', 'bright', 'contrast', 'sat', 'dark', 'ring', 'rg', 'bg', 'lap', 'p5', 'p95', 'block']
FI = {n: i for i, n in enumerate(FEATURES)}

# degradations for the demo (specs use the registry in dr_ops.py)
DEMO_DEGRADATIONS = {
    'lowlight':     [('lowlight', dict(dark=0.25))],
    'haze':         [('haze', dict(t=0.6))],
    'noise':        [('pg_noise', dict(gain=1))],
    'underexposed': [('gamma', dict(g=3.0))],
    'mixed':        [('pg_noise', dict(gain=1)), ('vignette', dict(strength=0.9)), ('jpeg', dict(q=40))],
}


class Classifier:
    """Locked EfficientNet-B0 (binary: referable DR = grade >= 2). Call with a list of 512x512x3 uint8 RGB arrays."""
    def __init__(self, weights, device=None):
        import torch, dr_common
        self.torch = torch
        self.dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        meta = json.load(open(weights / 'locked_model.json'))
        self.thr = float(meta['threshold'])                       # locked operating point (sens >= 90% on APTOS-val)
        self.model = dr_common.build_model(pretrained=False).to(self.dev)
        self.model.load_state_dict(torch.load(weights / 'locked_model.pt', map_location=self.dev)); self.model.eval()
        self.mean = torch.tensor(dr_common.MEAN, device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor(dr_common.STD, device=self.dev).view(1, 3, 1, 1)

    def __call__(self, arrs, bs=16):
        t, out = self.torch, []
        with t.no_grad():
            for i in range(0, len(arrs), bs):
                x = t.from_numpy(np.stack(arrs[i:i + bs])).to(self.dev).permute(0, 3, 1, 2).float() / 255
                out.append(t.sigmoid(self.model(((x - self.mean) / self.std)).squeeze(1)).float().cpu())
        return t.cat(out).numpy()


# ------------------------------------------------------------------ policies (all take st13 = 12 statistics + p0)
def rule_choice(st, P):
    """Threshold checklist: gate limits are the 1%/99% quantiles of CLEAN images (P['lo'], P['hi'])."""
    lo, hi = P['lo'], P['hi']
    noisy = st[FI['noise']] > hi[FI['noise']]
    light = (st[FI['bright']] < lo[FI['bright']]) or (st[FI['bright']] > hi[FI['bright']]) \
        or (st[FI['contrast']] < lo[FI['contrast']]) or (st[FI['dark']] > hi[FI['dark']])
    return 4 if noisy and light else 3 if noisy else 1 if light else 0


def selector_choice(st, P):
    """6 gradient-boosting regressors predict the cost of each library pipeline; keep 'none' unless another is clearly cheaper."""
    pred = np.array([m.predict(st[None, :])[0] for m in P['selector']])
    return int(pred.argmin()) if pred.min() + P['TOL'] < pred[0] else 0


def _sur(models, X):
    p = np.stack([m.predict(X) for m in models]); return p.mean(0), p.std(0)     # ensemble mean / disagreement


def cma_choice(st, P, seed=0):
    """CMA-ES over the continuous action [gamma, dehaze, denoise, order]; objective = predicted cost + LAM * ensemble std."""
    import cma
    sur, LAM, TOL = P['surrogate'], P['LAM'], P['TOL']
    def f(A): mu, sd = _sur(sur, np.c_[np.repeat(st[None, :], len(A), 0), A]); return (mu + LAM * sd).tolist()
    es = cma.CMAEvolutionStrategy([0.4, 0.4, 0.4, 0.5], 0.3, {'bounds': [0, 1], 'popsize': 10, 'maxiter': 10, 'verbose': -9, 'seed': seed + 1})
    while not es.stop():
        A = es.ask(); es.tell(A, f(np.array(A)))
    best = np.clip(es.result.xbest, 0, 1)
    m0, s0 = _sur(sur, np.r_[st, np.zeros(4)][None, :]); mb, sb = _sur(sur, np.r_[st, best][None, :])
    return best if (mb + LAM * sb)[0] + TOL < (m0 + LAM * s0)[0] else np.zeros(4)      # restore only if predicted gain > TOL


def choose_action(policy, st, P, seed):
    if policy == 'none': return np.zeros(4), 'none'
    if policy == 'rule': c = rule_choice(st, P); return A_LIB[c], LIB_NAMES[c]
    if policy == 'selector': c = selector_choice(st, P); return A_LIB[c], LIB_NAMES[c]
    a = cma_choice(st, P, seed)
    return a, ('none' if a.max() < 0.05 else f'gamma={a[0]:.2f} dehaze={a[1]:.2f} denoise={a[2]:.2f} denoise_{"first" if a[3] <= 0.5 else "last"}')


# ------------------------------------------------------------------ main
def list_images(p):
    p = Path(p)
    return [p] if p.is_file() else sorted(f for f in p.rglob('*') if f.suffix.lower() in IMG_EXT)


def name_of(v): return 'referable' if v else 'non-referable'


def reference_from_name(stem):
    """Clean-image decision encoded in the demo file names (no clean copy is shipped for the pre-degraded files).
    clean_referable_* / clean_healthy_* : the file itself is the clean image ('self');
    <degradation>_<id>.jpg              : degraded copy of an image the classifier calls referable when clean (chosen that way when the demo set was built)."""
    if stem.startswith('clean_'): return 'self'
    if any(stem.startswith(d + '_') for d in DEMO_DEGRADATIONS): return True
    return None


def judge(c, b, f, degraded=True):
    """c = decision on the clean image, b = on the (degraded) input, f = after restoration. Returns (status, sentence)."""
    if c is None:
        return 'NO REFERENCE', f'input {name_of(b)} -> restored {name_of(f)} ' + ('(decision changed; no clean reference for this file)' if b != f else '(unchanged; no clean reference for this file)')
    if c == b and not degraded:                                  # a clean image was given: restoration should leave the decision alone
        return (('NO CHANGE NEEDED', f'clean image, {name_of(c)}; restoration left it {name_of(f)}') if f == c else
                ('HARMED', f'clean image, {name_of(c)}; restoration changed it to {name_of(f)}'))
    if c == b:
        return (('NO CHANGE NEEDED', f'clean {name_of(c)}; the degradation did not change the decision; still {name_of(f)}') if f == c else
                ('HARMED', f'clean {name_of(c)}; stayed {name_of(b)} after degradation; restoration made it {name_of(f)}'))
    return (('RESCUED', f'clean {name_of(c)} -> degraded {name_of(b)} -> restored {name_of(f)} again') if f == c else
            ('NOT RESCUED', f'clean {name_of(c)} -> degraded {name_of(b)} -> still {name_of(f)} after restoration'))



def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, help='image file or folder of fundus images')
    ap.add_argument('--weights', default=str(HERE / 'weights'))
    ap.add_argument('--policy', default='cma', choices=['cma', 'selector', 'rule', 'none'])
    ap.add_argument('--degrade', default=None, choices=list(DEMO_DEGRADATIONS) + ['all'], help="simulate a poor upload before restoring ('all' = every demo degradation)")
    ap.add_argument('--out', default='sample_output/results.csv')
    ap.add_argument('--save-restored', default=None, help='folder to write the restored 512 px images')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default=None)
    a = ap.parse_args(argv)

    W = Path(a.weights)
    os.environ['DR_REF_PATH'] = str(W / 'ref_cdf.npy')           # clean colour reference used by gamma_auto (read by dr_ops)
    import dr_ops, dr_rl
    from dr_common import crop_square
    import joblib
    clf = Classifier(W, a.device); thr = clf.thr
    P = joblib.load(W / 'policy.joblib') if a.policy != 'none' else None
    paths = list_images(a.input); assert paths, f'no images found in {a.input}'
    if a.save_restored: Path(a.save_restored).mkdir(parents=True, exist_ok=True)

    degs = list(DEMO_DEGRADATIONS) if a.degrade == 'all' else [a.degrade]
    rows = []
    for path in paths:
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        x_clean = cv2.resize(crop_square(rgb), (512, 512), interpolation=cv2.INTER_AREA)   # same geometry as training masters
        p_clean = float(clf([x_clean])[0]) if a.degrade else None                         # reference: decision on the undegraded image
        for d in degs:
            t0 = time.time(); x = x_clean.copy()
            if d:
                rng = np.random.default_rng(a.seed)
                for name, kw in DEMO_DEGRADATIONS[d]: x = dr_ops.DEG[name](x, rng, **kw)
                x = dr_rl.to512(x)
            p0 = float(clf([x])[0])
            st = np.r_[dr_rl.state_features(x), p0].astype(np.float32)                   # 12 statistics + p0
            act, desc = choose_action(a.policy, st, P, a.seed)
            y = dr_rl.restore(x, act) if act.max() >= 0.05 else x
            p1 = float(clf([y])[0]) if y is not x else p0
            if a.save_restored: cv2.imwrite(str(Path(a.save_restored) / f"{path.stem}{'_' + d if d else ''}_restored.png"), cv2.cvtColor(y, cv2.COLOR_RGB2BGR))
            if p_clean is not None: c, ref = p_clean >= thr, 'computed on the clean image'
            else:
                k = reference_from_name(path.stem)
                c, ref = ((p0 >= thr) if k == 'self' else k), ('this file is the clean image' if k == 'self' else 'from file name' if k is not None else 'none')
            status, sentence = judge(c, p0 >= thr, p1 >= thr, degraded=bool(d) or ref == 'from file name')
            rows.append(dict(image=path.name, degradation=d or '', policy=a.policy, status=status, verdict=sentence, action=desc,
                             p_clean=None if p_clean is None else round(p_clean, 4), referable_clean=c, clean_reference=ref,
                             p_before=round(p0, 4), referable_before=p0 >= thr, p_after=round(p1, 4), referable_after=p1 >= thr,
                             decision_changed=(p0 >= thr) != (p1 >= thr), seconds=round(time.time() - t0, 2)))
            pc = f'p={p_clean:.3f}' if p_clean is not None else ref
            print(f"{path.name}" + (f"  [degradation: {d}]" if d else '') + f"\n   [{status}] {sentence}\n   probabilities: clean {pc} | input {p0:.3f} | restored {p1:.3f} "
                  f"(locked threshold {thr:.3f}) | action: {desc}", flush=True)
    df = pd.DataFrame(rows); Path(a.out).parent.mkdir(parents=True, exist_ok=True); df.to_csv(a.out, index=False)
    n = df.status.value_counts(); g = lambda k: int(n.get(k, 0))
    print(f"\nwrote {a.out}  |  policy: {a.policy}  |  threshold {thr:.4f}")
    print(f"SUMMARY over {len(df)} runs: rescued {g('RESCUED')} | not rescued {g('NOT RESCUED')} | no change needed {g('NO CHANGE NEEDED')} "
          f"| harmed by restoration {g('HARMED')} | no clean reference {g('NO REFERENCE')}")
    if g('RESCUED') + g('NOT RESCUED'): print(f"degradation caused a wrong decision in {g('RESCUED') + g('NOT RESCUED')} runs; restoration fixed {g('RESCUED')} of them")
    return df


if __name__ == '__main__':
    main()