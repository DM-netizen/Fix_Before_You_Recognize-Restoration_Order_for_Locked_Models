# LOCKED pipeline = locked_preprocess + model + threshold.
# Degradation / restoration plug in via `raw_transform` (a picklable callable acting on the
# full master RGB image BEFORE the locked preprocessing).
import cv2, numpy as np, torch, timm
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, roc_curve

MASTER_SIZE = 1024      # cached clean master images (square, black-border cropped)
IMG_SIZE = 512          # input size of the locked classifier
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def crop_square(rgb, size=MASTER_SIZE, thresh=10):
    mask = rgb.mean(axis=2) > thresh
    if mask.sum() > 100:
        ys, xs = np.where(mask)
        rgb = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = rgb.shape[:2]; s = max(h, w)
    canvas = np.zeros((s, s, 3), np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = rgb
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def locked_preprocess(rgb):
    x = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = (x.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class DRDataset(Dataset):
    def __init__(self, df, augment=False, raw_transform=None):
        self.df = df.reset_index(drop=True); self.augment = augment; self.raw_transform = raw_transform

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        rgb = cv2.cvtColor(cv2.imread(r['master_path']), cv2.COLOR_BGR2RGB)
        if self.raw_transform is not None: rgb = self.raw_transform(rgb)
        x = locked_preprocess(rgb)
        if self.augment:   # geometric only, on purpose: photometric aug would pre-train robustness
            if np.random.rand() < 0.5: x = torch.flip(x, [2])
            if np.random.rand() < 0.5: x = torch.flip(x, [1])
            x = torch.rot90(x, np.random.randint(4), [1, 2])
        return x, torch.tensor(float(r['label'])), i


def build_model(pretrained=True):
    return timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); out = []
    for x, _, _ in loader:
        out.append(torch.sigmoid(model(x.to(device)).squeeze(1)).float().cpu())
    return torch.cat(out).numpy()


def pick_threshold(y, p, min_sens=0.90):
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= min_sens)[0]
    return float(thr[ok[np.argmin(fpr[ok])]])


def metrics(y, p, thr):
    y = np.asarray(y); pred = p >= thr
    tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    return dict(n=len(y), auc=roc_auc_score(y, p), acc=(pred == y).mean(),
                sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1))
