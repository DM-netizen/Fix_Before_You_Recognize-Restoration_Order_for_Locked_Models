# Degradation (DEG) and restoration (RES) operations + picklable SpecFn.
# Degradation spec : list of (name, kwargs) applied in order, e.g. [('gauss_noise', {'sigma': 8}), ('jpeg', {'q': 50})]
# Restoration chain: list of (name, kwargs) applied in order after the degradation.
# Specs are plain data (JSON-able), so they can serve as RL actions and survive multiprocessing 'spawn'.
import io, os, functools
import numpy as np, cv2
from PIL import Image

DEG, RES = {}, {}
def deg(name):
    def d(f): DEG[name] = f; return f
    return d
def res(name):
    def d(f): RES[name] = f; return f
    return d

def _clip(x): return np.clip(x, 0, 255).astype(np.uint8)
def _grid(h, w):
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    return (yy - h / 2) / (h / 2), (xx - w / 2) / (w / 2)

# ============================ degradations: f(x, rng, **kw) ============================
@deg('gauss_blur')
def d_gauss_blur(x, rng, sigma): return cv2.GaussianBlur(x, (0, 0), sigma)
@deg('defocus')
def d_defocus(x, rng, radius):
    k = np.zeros((2 * radius + 1,) * 2, np.float32); cv2.circle(k, (radius, radius), radius, 1, -1)
    return cv2.filter2D(x, -1, k / k.sum())
@deg('motion')
def d_motion(x, rng, length):
    k = np.zeros((length, length), np.float32); cv2.line(k, (0, length // 2), (length - 1, length // 2), 1, 1)
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), rng.uniform(0, 180), 1)
    k = cv2.warpAffine(k, M, (length, length))
    return cv2.filter2D(x, -1, k / max(k.sum(), 1e-6))

@deg('gauss_noise')
def d_gauss_noise(x, rng, sigma): return _clip(x + rng.normal(0, sigma, x.shape))
@deg('pg_noise')
def d_pg_noise(x, rng, gain, read=2.0):          # shot noise (variance grows with brightness) + read noise
    return _clip(rng.poisson(np.clip(x.astype(np.float32), 0, 255) / gain) * gain + rng.normal(0, read, x.shape))
@deg('lowlight')
def d_lowlight(x, rng, dark, gain=1.0, read=2.0):  # underexposed capture, noise, then brightened
    d = x.astype(np.float32) * dark
    return _clip((rng.poisson(d / gain) * gain + rng.normal(0, read, x.shape)) / dark)
@deg('chroma_noise')
def d_chroma_noise(x, rng, sigma):                # colour-only noise
    y = cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y[..., 1:] += rng.normal(0, sigma, y[..., 1:].shape)
    return cv2.cvtColor(_clip(y), cv2.COLOR_YCrCb2RGB)
@deg('saltpepper')
def d_saltpepper(x, rng, frac):                   # hot / dead pixels
    m = rng.random(x.shape[:2]); y = x.copy(); y[m < frac / 2] = 0; y[m > 1 - frac / 2] = 255; return y

@deg('jpeg')
def d_jpeg(x, rng, q, sub=2):                     # sub: 0 = 4:4:4, 2 = 4:2:0 chroma subsampling
    b = io.BytesIO(); Image.fromarray(x).save(b, 'JPEG', quality=int(q), subsampling=sub); b.seek(0)
    return np.array(Image.open(b).convert('RGB'))
@deg('resize')
def d_resize(x, rng, scale):
    h, w = x.shape[:2]; return cv2.resize(x, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
@deg('recompress')
def d_recompress(x, rng, rounds, q, scale=0.9):   # forwarded repeatedly: resize + JPEG each round
    for _ in range(rounds): x = d_jpeg(d_resize(x, rng, scale), rng, q)
    return x

@deg('vignette')
def d_vignette(x, rng, strength):
    yy, xx = _grid(*x.shape[:2]); r = np.minimum(np.hypot(yy, xx), 1)
    return _clip(x * (1 - strength * r ** 2)[..., None])
@deg('uneven')
def d_uneven(x, rng, strength):                   # one side brighter than the other
    yy, xx = _grid(*x.shape[:2]); a = rng.uniform(0, 2 * np.pi)
    return _clip(x * (1 + strength * (np.cos(a) * xx + np.sin(a) * yy) / 1.42)[..., None])
@deg('gamma')
def d_gamma(x, rng, g): return _clip(255 * (x / 255.0) ** g)      # g<1 brighter, g>1 darker
@deg('contrast')
def d_contrast(x, rng, c): m = x.mean(); return _clip((x - m) * c + m)
@deg('haze')
def d_haze(x, rng, t, A=210): return _clip(x * (1 - t) + A * t)
@deg('wb')
def d_wb(x, rng, r, g, b): return _clip(x * np.array([r, g, b], np.float32))
@deg('saturation')
def d_saturation(x, rng, s):
    h = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.float32); h[..., 1] *= s
    return cv2.cvtColor(_clip(h), cv2.COLOR_HSV2RGB)
@deg('posterize')
def d_posterize(x, rng, bits): s = 8 - bits; return (x >> s) << s

@deg('smooth')
def d_smooth(x, rng, passes):                     # aggressive phone "beauty"/denoise smoothing
    for _ in range(passes): x = cv2.bilateralFilter(x, 9, 40, 9)
    return x
@deg('oversharp')
def d_oversharp(x, rng, amt, s=2): return cv2.addWeighted(x, 1 + amt, cv2.GaussianBlur(x, (0, 0), s), -amt, 0)

@deg('glare')
def d_glare(x, rng, strength, radius=0.18):
    h, w = x.shape[:2]; cy, cx = rng.uniform(0.25, 0.75, 2) * np.array([h, w]); yy, xx = np.mgrid[:h, :w]
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (radius * h) ** 2))
    return _clip(x + 255 * strength * g[..., None])
@deg('eyelid')
def d_eyelid(x, rng, frac):                       # soft dark band over top or bottom
    h = x.shape[0]; yy = np.arange(h, dtype=np.float32)[:, None]
    d = (frac * h - yy) if rng.random() < 0.5 else (yy - (1 - frac) * h)
    return _clip(x * (1 - np.clip(d / (0.03 * h) + 0.5, 0, 1))[:, :, None])
@deg('zoom')
def d_zoom(x, rng, z):                            # tight field of view (periphery lost)
    h, w = x.shape[:2]; ch, cw = int(h / z), int(w / z); y0, x0 = (h - ch) // 2, (w - cw) // 2
    return cv2.resize(x[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_CUBIC)
@deg('shift')
def d_shift(x, rng, frac):                        # off-centre capture
    h, w = x.shape[:2]; a = rng.uniform(0, 2 * np.pi)
    return cv2.warpAffine(x, np.float32([[1, 0, frac * w * np.cos(a)], [0, 1, frac * h * np.sin(a)]]), (w, h))
@deg('rotate')
def d_rotate(x, rng, deg):                        # handheld tilt
    h, w = x.shape[:2]
    return cv2.warpAffine(x, cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-deg, deg), 1), (w, h))

# ============================ restoration: f(x, **kw), deterministic ============================
@functools.lru_cache(maxsize=8)
def _disc(h, w, r=0.42):
    yy, xx = np.mgrid[:h, :w]
    return ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) < (r * min(h, w)) ** 2

def _cdf(ch, r=0.42):
    c = np.cumsum(np.bincount(ch[_disc(*ch.shape, r)], minlength=256)).astype(np.float64)
    return c / c[-1]

_REF = {}
def build_ref(paths, out):
    # reference colour distribution = mean per-channel CDF over CLEAN images (set DR_REF_PATH to `out`)
    acc = np.zeros((3, 256))
    for p in paths:
        im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        for c in range(3): acc[c] += _cdf(im[..., c])
    acc /= len(paths); np.save(out, acc); _REF.clear(); return acc
def ref():
    if 'cdf' not in _REF: _REF['cdf'] = np.load(os.environ['DR_REF_PATH'])
    return _REF['cdf']
def _ref_means():
    R = ref(); pmf = np.diff(np.concatenate([np.zeros((3, 1)), R], axis=1), axis=1)
    return (pmf * np.arange(256)).sum(1)

@res('nlm')
def r_nlm(x, h=10):
    return cv2.cvtColor(cv2.fastNlMeansDenoisingColored(cv2.cvtColor(x, cv2.COLOR_RGB2BGR), None, h, h, 7, 11), cv2.COLOR_BGR2RGB)
@res('gauss')
def r_gauss(x, sigma=1.0): return cv2.GaussianBlur(x, (0, 0), sigma)
@res('median')
def r_median(x, k=3): return cv2.medianBlur(x, k)
@res('bilateral')
def r_bilateral(x, sc=40): return cv2.bilateralFilter(x, 9, sc, 5)
@res('wavelet')
def r_wavelet(x, sigma=10):
    from skimage.restoration import denoise_wavelet
    out = denoise_wavelet(x.astype(np.float32) / 255, sigma=sigma / 255, channel_axis=-1, convert2ycbcr=True,
                          method='BayesShrink', mode='soft')
    return _clip(out * 255)
@res('unsharp')
def r_unsharp(x, amt=0.5, s=2): return cv2.addWeighted(x, 1 + amt, cv2.GaussianBlur(x, (0, 0), s), -amt, 0)
@res('wiener')
def r_wiener(x, sigma=2.0, balance=0.05):
    from skimage.restoration import wiener
    k = (int(6 * sigma) | 1); g = cv2.getGaussianKernel(k, sigma); psf = g @ g.T
    f = x.astype(np.float32) / 255
    return _clip(np.stack([wiener(f[..., c], psf, balance) for c in range(3)], -1) * 255)
@res('hist_match')
def r_hist_match(x, alpha=1.0):                   # map each channel's histogram to the clean-reference one
    R = ref(); out = np.empty_like(x)
    for c in range(3):
        lut = np.clip(np.searchsorted(R[c], _cdf(x[..., c])), 0, 255).astype(np.uint8)
        out[..., c] = lut[x[..., c]]
    return cv2.addWeighted(x, 1 - alpha, out, alpha, 0) if alpha < 1 else out
@res('clahe')
def r_clahe(x, clip=2.0, tile=8):
    lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
@res('gamma_auto')
def r_gamma_auto(x, strength=1.0):                # pick gamma so mean brightness matches the clean reference
    target = float(_ref_means().mean()) / 255
    cur = float(x[_disc(*x.shape[:2])].mean()) / 255
    if cur <= 1e-3 or cur >= 0.999: return x
    g = float(np.clip(np.log(target) / np.log(cur), 0.4, 2.5)) ** strength
    return _clip(255 * (x / 255.0) ** g)
@res('color_match')
def r_color_match(x, strength=1.0):               # per-channel gain so channel means match the clean reference
    cur = x[_disc(*x.shape[:2])].reshape(-1, 3).mean(0)
    gain = (_ref_means() / np.maximum(cur, 1.0)) ** strength
    return _clip(x * gain.astype(np.float32))
@res('dehaze')
def r_dehaze(x, omega=0.8, t0=0.15, patch=15):    # dark-channel-prior dehazing
    I = x.astype(np.float32) / 255; ker = np.ones((patch, patch), np.uint8)
    mk = _disc(*x.shape[:2])
    dark = cv2.erode(I.min(axis=2), ker)
    n = max(int(mk.sum() * 0.001), 1)
    idx = np.argpartition(np.where(mk, dark, -1).ravel(), -n)[-n:]
    A = I.reshape(-1, 3)[idx].mean(0)
    t = 1 - omega * cv2.erode((I / np.maximum(A, 1e-3)).min(axis=2), ker)
    t = cv2.GaussianBlur(np.clip(t, t0, 1), (0, 0), patch / 3)
    return _clip(((I - A) / t[..., None] + A) * 255)
@res('flatten')
def r_flatten(x, sigma=60, alpha=1.0):            # remove slow illumination gradients (masked single-scale retinex)
    f = x.astype(np.float32); mk = _disc(*x.shape[:2], 0.5); m3 = mk.astype(np.float32)[..., None]
    bg = cv2.GaussianBlur(f * m3, (0, 0), sigma) / (cv2.GaussianBlur(m3[..., 0], (0, 0), sigma)[..., None] + 1e-3)
    gm = (bg * m3).reshape(-1, 3).sum(0) / m3.sum()
    y = _clip(f / (bg + 1.0) * gm * m3)
    return cv2.addWeighted(x, 1 - alpha, y, alpha, 0) if alpha < 1 else y

# ============================ picklable transform ============================
class SpecFn:
    # callable: degrade (seeded by clean-image content, so reproducible) then restore
    def __init__(self, deg=None, chain=None):
        self.deg = [tuple(s) for s in (deg or [])]; self.chain = [tuple(s) for s in (chain or [])]
    def __call__(self, rgb):
        if self.deg:
            rng = np.random.default_rng(int(rgb[::97, ::97].sum()) % (2 ** 32))
            for name, kw in self.deg: rgb = DEG[name](rgb, rng, **kw)
        for name, kw in self.chain: rgb = RES[name](rgb, **kw)
        return rgb
