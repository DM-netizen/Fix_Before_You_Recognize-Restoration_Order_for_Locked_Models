# Worker-safe module (no torch): random degradation sampler, continuous-action restorer, state features, sample worker.
# Everything works at 512 px (the classifier's input size). Noise / blur ranges are scaled so that they roughly match the
# earlier 1024-px-then-downsample experiments (downsampling by 2 averages away part of the noise and halves blur radii).
import os, numpy as np, cv2
import dr_ops

FAMILIES = ['light', 'blur', 'noise', 'color', 'sharp', 'compress']
def _lu(r, a, b): return float(np.exp(r.uniform(np.log(a), np.log(b))))
COMP = {   # name -> (family, sampler of kwargs)
 'gamma':       ('light',    lambda r: dict(g=_lu(r, 0.5, 3.0))),
 'haze':        ('light',    lambda r: dict(t=r.uniform(0.1, 0.7))),
 'contrast':    ('light',    lambda r: dict(c=r.uniform(0.25, 0.8))),
 'vignette':    ('light',    lambda r: dict(strength=r.uniform(0.3, 0.9))),
 'gauss_blur':  ('blur',     lambda r: dict(sigma=r.uniform(0.5, 2.5))),
 'defocus':     ('blur',     lambda r: dict(radius=int(r.integers(1, 5)))),
 'motion':      ('blur',     lambda r: dict(length=int(r.integers(6, 22)))),
 'pg_noise':    ('noise',    lambda r: dict(gain=_lu(r, 0.1, 1.5))),
 'gauss_noise': ('noise',    lambda r: dict(sigma=r.uniform(2, 14))),
 'lowlight':    ('noise',    lambda r: dict(dark=r.uniform(0.25, 0.7), gain=0.25)),
 'saturation':  ('color',    lambda r: dict(s=r.uniform(0.2, 0.8))),
 'wb':          ('color',    lambda r: dict(r=r.uniform(0.8, 1.25), g=1.0, b=r.uniform(0.8, 1.25))),
 'oversharp':   ('sharp',    lambda r: dict(amt=r.uniform(1.0, 3.0))),
 'smooth':      ('sharp',    lambda r: dict(passes=int(r.integers(1, 4)))),
 'jpeg':        ('compress', lambda r: dict(q=int(r.integers(10, 71)))),
 'resize':      ('compress', lambda r: dict(scale=r.uniform(0.35, 0.8))),
}
def _py(kw): return {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v) for k, v in kw.items()}
def families(spec): return {COMP[n][0] for n, _ in spec}

def sample_spec(rng, p_clean=0.3, p_comp=(0.55, 0.30, 0.15)):
    if rng.random() < p_clean: return []
    k = int(rng.choice([1, 2, 3], p=p_comp))
    fams = [FAMILIES[i] for i in rng.choice(len(FAMILIES), size=k, replace=False)]
    comps = []
    for f in fams:
        names = [n for n, (ff, _) in COMP.items() if ff == f]; n = names[int(rng.integers(len(names)))]
        comps.append((n, _py(COMP[n][1](rng))))
    cap = [c for c in comps if COMP[c[0]][0] in ('light', 'blur', 'noise', 'color')]          # capture stage, random order
    cap = [cap[i] for i in rng.permutation(len(cap))]
    return cap + [c for c in comps if COMP[c[0]][0] == 'sharp'] + [c for c in comps if COMP[c[0]][0] == 'compress']   # compression last

def random_actions(rng, n, p_on=0.6):
    A = rng.uniform(0.1, 1.0, size=(n, 4)); A[:, :3] *= (rng.random((n, 3)) < p_on); A[:, 3] = rng.random(n); return A

# ---- restorer: a = [gamma strength, dehaze strength, denoise strength, order flag], all in [0, 1]; strength < 0.05 = skip ----
def _nlm(x, h, tw=5, sw=9):
    return cv2.cvtColor(cv2.fastNlMeansDenoisingColored(cv2.cvtColor(x, cv2.COLOR_RGB2BGR), None, h, h, tw, sw), cv2.COLOR_BGR2RGB)
def restore(img, a):
    a = np.clip(np.asarray(a, float), 0, 1)
    den = (lambda x: _nlm(x, float(20 * a[2]))) if a[2] >= 0.05 else (lambda x: x)
    if a[3] <= 0.5: img = den(img)                                   # order flag: denoise first ...
    if a[0] >= 0.05: img = dr_ops.RES['gamma_auto'](img, strength=float(a[0]))
    if a[1] >= 0.05: img = dr_ops.RES['dehaze'](img, omega=float(0.9 * a[1]), patch=9)
    if a[3] > 0.5: img = den(img)                                    # ... or denoise last
    return img

# ---- state features of the (degraded) image at 512 px ----
_YY, _XX = np.mgrid[:512, :512]; _R = np.hypot(_YY - 256, _XX - 256) / 512
MASK, RING, CENT = _R < 0.45, (_R > 0.30) & (_R < 0.45), _R < 0.15
_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
FEATURE_NAMES = ['noise', 'bright', 'contrast', 'sat', 'dark', 'ring', 'rg', 'bg', 'lap', 'p5', 'p95', 'block']
def to512(x): return x if x.shape[:2] == (512, 512) else cv2.resize(x, (512, 512), interpolation=cv2.INTER_AREA)
def state_features(s):
    g = s[..., 1].astype(np.float32); gray = s.mean(axis=2); R, G, B = [s[..., i].astype(np.float32) for i in range(3)]
    d = np.abs(np.diff(gray, axis=1)); edge = (np.arange(d.shape[1]) % 8 == 7)
    return np.array([np.median(np.abs(cv2.filter2D(g, -1, _K))[MASK]) / 0.6745 / 6, gray[MASK].mean() / 255, gray[MASK].std() / 255,
                     cv2.cvtColor(s, cv2.COLOR_RGB2HSV)[..., 1][MASK].mean() / 255, s.min(axis=2)[MASK].mean() / 255,
                     gray[RING].mean() / (gray[CENT].mean() + 1), R[MASK].mean() / (G[MASK].mean() + 1), B[MASK].mean() / (G[MASK].mean() + 1),
                     float(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)[MASK].var()), np.percentile(gray[MASK], 5) / 255,
                     np.percentile(gray[MASK], 95) / 255, d[:, edge].mean() / (d[:, ~edge].mean() + 1e-3)], np.float32)

def init_worker():                    # keep the laptop usable: lower the priority of worker processes
    try:
        if os.name == 'nt':
            import ctypes; ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
        else: os.nice(10)
    except Exception: pass

def work(task):                       # (path of 512 master, degradation spec, seed, list of action vectors) -> degraded + restored images
    path, spec, seed, actions = task
    deg = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB); rng = np.random.default_rng(seed)
    for name, kw in spec: deg = dr_ops.DEG[name](deg, rng, **kw)
    return dict(state=state_features(to512(deg)), deg=to512(deg), outs=[to512(restore(deg, a)) for a in actions])
