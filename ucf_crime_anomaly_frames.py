"""
UCF-Crime anomaly detection from the Kaggle FRAME dataset (odins0n/ucf-crime-dataset)
3D CNN (Kinetics-pretrained, frozen) + BiGRU + MIL ranking loss, video-level evaluation

Self-contained: this one file is all you need (no other script required).

Differences from the raw-video pipeline
  * Input is folders of extracted frames: Train/<ClassName>/<video>_<frame>.png (same for Test).
    The folder name gives the video-level label (NormalVideos = normal, everything else = anomaly).
  * There are no temporal annotations, so only VIDEO-level metrics are possible
    (AUC, average precision, accuracy, per-class detection rate) - not frame-level AUC.
  * Frames are low-resolution and sparsely sampled, so accuracy will be lower than with raw videos.

Usage
    pip install torch torchvision opencv-python scikit-learn matplotlib numpy
    python ucf_crime_anomaly_frames.py --root /path/to/ucf-crime-dataset --stage inspect   # print the structure
    python ucf_crime_anomaly_frames.py --root /path/to/ucf-crime-dataset --stage all       # features + train + evaluate

Outputs (./outputs_frames): best_model.pt, training_curves.png, roc_curve.png,
                            class_detection.png, score_timelines.png, printed metrics
"""
import argparse
import os
import re
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import models

cv2.setNumThreads(0)
# FIXED: This regular expression correctly parses <video>_<number>.png
FRAME_RE = re.compile(r"^(?P<vid>.*)_(?P<idx>\d+)\.png$", re.I)
SIZE = 112
FEAT_DIM = 512
MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)


# ----------------------------------------------------------------------------- model (self-contained)
def to_bag(feats, T):
    """Mean-pool a variable-length clip sequence into T segments, then L2-normalise."""
    f = feats.astype(np.float32)
    edges = np.linspace(0, len(f), T + 1).astype(int)
    out = np.empty((T, f.shape[1]), np.float32)
    for s in range(T):
        a = min(edges[s], len(f) - 1)
        b = max(edges[s + 1], a + 1)
        out[s] = f[a:b].mean(0)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-8)


class GRUAnomaly(nn.Module):
    """3D-CNN clip features -> projection -> BiGRU -> per-segment anomaly score in [0,1]."""

    def __init__(self, d_in=FEAT_DIM, d_proj=256, d_h=128):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d_proj), nn.ReLU(), nn.Dropout(0.3))
        self.gru = nn.GRU(d_proj, d_h, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * d_h, 64), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):                      # x: (B, T, D)
        h, _ = self.gru(self.proj(x))
        return self.head(h).squeeze(-1)        # (B, T)


def mil_loss(sa, sn, l_smooth=8e-5, l_sparse=8e-5):
    rank = torch.relu(1.0 - sa.max(1)[0] + sn.max(1)[0]).mean()
    smooth = ((sa[:, 1:] - sa[:, :-1]) ** 2).sum(1).mean()
    sparse = sa.sum(1).mean()
    return rank + l_smooth * smooth + l_sparse * sparse


@torch.no_grad()
def score(model, X, dev, bs=512):
    model.eval()
    return torch.cat([model(X[i:i + bs].to(dev)).cpu() for i in range(0, len(X), bs)]).numpy()


# ----------------------------------------------------------------------------- args / indexing
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="folder containing the downloaded Train/ and Test/ folders")
    p.add_argument("--stage", default="all", choices=["inspect", "extract", "train", "all"])
    p.add_argument("--feat_dir", default="features_frames")
    p.add_argument("--out", default="outputs_frames")
    p.add_argument("--clip_len", type=int, default=16, help="consecutive sampled frames per clip")
    p.add_argument("--max_clips", type=int, default=64, help="clips per video (uniformly spread)")
    p.add_argument("--gpu_chunk", type=int, default=64)
    p.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 2))
    p.add_argument("--T", type=int, default=32, help="segments per video bag")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--B", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_per_class", type=int, default=0,
                   help="train videos per anomaly class (normals matched); 0 = all videos")
    p.add_argument("--test_per_class", type=int, default=0, help="same for test; 0 = all videos")
    p.add_argument("--prefer_small", action=argparse.BooleanOptionalAction, default=True,
                   help="when subsetting, pick videos with the fewest frames (fastest)")
    p.add_argument("--chunk_frames", type=int, default=-1,
                   help="split each parsed 'video' into consecutive chunks of this many frames. "
                        "-1 = automatic (only if the frames turn out to be combined per class, "
                        "then 640), 0 = never, N = always")
    return p.parse_args()


def is_anom(cls):
    return not cls.lower().startswith("normal")


def find_split_dir(root, name, max_depth=4):
    """Shallowest directory called `name` (case-insensitive) under root."""
    level = [root]
    for _ in range(max_depth):
        nxt = []
        for d in level:
            try:
                for e in os.scandir(d):
                    if e.is_dir():
                        if e.name.lower() == name:
                            return e.path
                        nxt.append(e.path)
            except OSError:
                pass
        level = nxt
    return None


def index_items(split_dir, split):
    vids = {}
    for dp, _, fs in os.walk(split_dir):
        cls = os.path.basename(dp)
        for f in fs:
            m = FRAME_RE.match(f)
            if m:
                vids.setdefault((cls, m["vid"]), []).append((int(m["idx"]), os.path.join(dp, f)))
    items = []
    for (cls, vid), fr in sorted(vids.items()):
        fr.sort()
        items.append({"name": f"{split}____{cls}___{vid}", "cls": cls, "vid": vid,
                      "paths": [p for _, p in fr]})
    return items


def subsample(items, per_class, rng, prefer_small):
    if per_class <= 0:
        return items
    groups = {}
    for it in items:
        groups.setdefault(it["cls"] if is_anom(it["cls"]) else "Normal", []).append(it)

    def pick(lst, k):
        if prefer_small:
            return sorted(lst, key=lambda it: len(it["paths"]))[:k]
        return [lst[i] for i in rng.permutation(len(lst))[:k]]

    anomaly_classes = [c for c in groups if c != "Normal"]
    out = []
    for c in anomaly_classes:
        out += pick(groups[c], per_class)
    out += pick(groups.get("Normal", []), per_class * len(anomaly_classes))
    return out


def needs_chunking(items):
    """True if frames look 'combined per class' (about one parsed video per class)."""
    n_cls = len({it["cls"] for it in items})
    return len(items) <= 2 * max(n_cls, 1)


def rechunk(items, chunk, clip_len):
    """Cut each item's frame list into consecutive chunks, each treated as one pseudo-video
    that inherits the class label (used when video identity is not in the file names)."""
    out = []
    for it in items:
        p = it["paths"]
        parts = [p[i:i + chunk] for i in range(0, len(p), chunk)]
        if len(parts) > 1 and len(parts[-1]) < 4 * clip_len:
            tail = parts.pop()
            parts[-1] = parts[-1] + tail
        for k, part in enumerate(parts):
            out.append({"name": f"{it['name']}_c{k:04d}", "cls": it["cls"],
                        "vid": f"{it['vid']}_c{k:04d}", "paths": part})
    return out


def inspect(train_items, test_items, root):
    print(f"\nDataset root: {root}")
    for tag, items in (("Train", train_items), ("Test", test_items)):
        classes = {}
        for it in items:
            classes.setdefault(it["cls"], []).append(len(it["paths"]))
        print(f"\n{tag}: {len(items)} videos, {sum(sum(v) for v in classes.values())} frames")
        for c, v in sorted(classes.items()):
            print(f"  {c:15s} {len(v):4d} videos | frames/video min {min(v)} median {int(np.median(v))} max {max(v)}")
        if items:
            print("  example frames:", [os.path.basename(p) for p in items[0]["paths"][:3]])
    if not train_items:
        print("\nNo frames matched the pattern <video>_<number>.png - tell me the file names above and "
              "I will adapt the parser.")


# ----------------------------------------------------------------------------- stage 1: features
class ClipDataset(Dataset):
    def __init__(self, items, a):
        self.items, self.a = items, a

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        a, it = self.a, self.items[i]
        paths, L = it["paths"], a.clip_len
        F = len(paths)
        n_clips = max(1, F // L)
        idx = np.arange(n_clips)
        if n_clips > a.max_clips:
            idx = np.linspace(0, n_clips - 1, a.max_clips).astype(int)
        frame_idx = np.minimum(idx[:, None] * L + np.arange(L)[None], F - 1).ravel()
        uniq, inv = np.unique(frame_idx, return_inverse=True)
        imgs = np.zeros((len(uniq), SIZE, SIZE, 3), np.uint8)
        for j, u in enumerate(uniq):
            im = cv2.imread(paths[u], cv2.IMREAD_COLOR)
            if im is None:
                continue
            if im.shape[0] != SIZE or im.shape[1] != SIZE:
                im = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            imgs[j] = im[:, :, ::-1]
        x = torch.from_numpy(imgs[inv]).view(len(idx), L, SIZE, SIZE, 3)
        return it["name"], x, F


@torch.no_grad()
def extract(args, items):
    os.makedirs(args.feat_dir, exist_ok=True)
    todo = [it for it in items if not os.path.exists(os.path.join(args.feat_dir, it["name"] + ".npz"))]
    if not todo:
        print("All features already cached.")
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = models.video.r3d_18(weights="DEFAULT")
    net.fc = nn.Identity()
    net = net.to(dev).eval().to(memory_format=torch.channels_last_3d)
    mean, std = MEAN.to(dev), STD.to(dev)
    dl = DataLoader(ClipDataset(todo, args), batch_size=None, shuffle=False, num_workers=args.workers,
                    pin_memory=dev.type == "cuda", prefetch_factor=2 if args.workers > 0 else None)
    t0 = time.time()
    for k, (name, x, n_frames) in enumerate(dl, 1):
        feats = []
        for c in x.split(args.gpu_chunk):
            c = c.to(dev, non_blocking=True).permute(0, 4, 1, 2, 3).float().div_(255)
            c = ((c - mean) / std).contiguous(memory_format=torch.channels_last_3d)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                feats.append(net(c).float().cpu())
        np.savez(os.path.join(args.feat_dir, name + ".npz"),
                 feats=torch.cat(feats).numpy().astype(np.float16), nframes=n_frames)
        if k % 50 == 0 or k == len(todo):
            print(f"  extracted {k}/{len(todo)} videos | {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- stage 2: train/eval
def load_bags(items, args):
    X, y, keep = [], [], []
    for it in items:
        p = os.path.join(args.feat_dir, it["name"] + ".npz")
        if os.path.exists(p):
            X.append(to_bag(np.load(p)["feats"], args.T))
            y.append(is_anom(it["cls"]))
            keep.append(it)
    return np.stack(X), np.array(y), keep


def train_and_eval(args, train_items, test_items):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    Xtr, ytr, _ = load_bags(train_items, args)
    Xte, yte, te_keep = load_bags(test_items, args)
    print(f"Train videos: {len(ytr)} (anomalous {ytr.sum()}, normal {(~ytr).sum()}) | Test videos: {len(yte)} "
          f"(anomalous {yte.sum()}, normal {(~yte).sum()})")

    val_mask = np.zeros(len(ytr), bool)
    for c in (True, False):
        idx = np.where(ytr == c)[0]
        val_mask[rng.choice(idx, max(1, int(len(idx) * args.val_frac)), replace=False)] = True
    Xt = torch.from_numpy(Xtr[~val_mask]).to(dev)
    yt = ytr[~val_mask]
    Xv, yv = torch.from_numpy(Xtr[val_mask]), ytr[val_mask]
    Xa, Xn = Xt[torch.from_numpy(yt).to(dev)], Xt[torch.from_numpy(~yt).to(dev)]

    model = GRUAnomaly().to(dev)
    print(f"Trainable parameters (GRU head): {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    steps = max(1, len(Xa) // args.B)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps)

    hist = {"loss": [], "val_auc": []}
    best, best_state = -1, None
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        pa = torch.randperm(len(Xa), device=dev)
        pn = torch.randperm(len(Xn), device=dev)
        tot = 0.0
        for s in range(steps):
            ia = pa[s * args.B:(s + 1) * args.B]
            inn = pn[s * args.B % len(Xn):][:len(ia)]
            if len(inn) < len(ia):
                inn = torch.randint(0, len(Xn), (len(ia),), device=dev)
            xa = Xa[ia] + 0.01 * torch.randn_like(Xa[ia])
            xn = Xn[inn] + 0.01 * torch.randn_like(Xn[inn])
            out = model(torch.cat([xa, xn]))
            loss = mil_loss(out[:len(xa)], out[len(xa):])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item()
        vauc = roc_auc_score(yv, score(model, Xv, dev).max(1))
        hist["loss"].append(tot / steps)
        hist["val_auc"].append(vauc)
        if vauc >= best:
            best, best_state = vauc, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == 1:
            print(f"Epoch {ep:03d}/{args.epochs} | loss {tot / steps:.4f} | val video-AUC {vauc:.4f}")
    print(f"Training time: {time.time() - t0:.1f}s | best val AUC {best:.4f}")
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "T": args.T}, os.path.join(args.out, "best_model.pt"))

    # threshold chosen on the validation videos only (Youden's J), then applied to the test set
    fv, tv, thv = roc_curve(yv, score(model, Xv, dev).max(1))
    thr = float(thv[np.argmax(tv - fv)])
    S = score(model, torch.from_numpy(Xte), dev)
    vs = S.max(1)
    pred = vs >= thr
    auc = roc_auc_score(yte, vs)
    ap = average_precision_score(yte, vs)
    acc = (pred == yte).mean()
    tp, fp = (pred & yte).sum(), (pred & ~yte).sum()
    fn, tn = (~pred & yte).sum(), (~pred & ~yte).sum()
    print(f"\nTEST video-level  AUC {auc:.4f} | AP {ap:.4f} | accuracy {acc:.4f} (threshold {thr:.3f} from validation)")
    print(f"  precision {tp / max(tp + fp, 1):.3f} | recall {tp / max(tp + fn, 1):.3f} | "
          f"false-alarm rate on normal videos {fp / max(fp + tn, 1):.3f}")

    classes = sorted({it["cls"] for it in te_keep}, key=lambda c: (is_anom(c) is False, c))
    rates, means = [], []
    print(f"\n{'class':15s} {'videos':>6s} {'mean max-score':>15s} {'flagged as anomaly':>20s}")
    for c in classes:
        m = np.array([it["cls"] == c for it in te_keep])
        rates.append(pred[m].mean())
        means.append(vs[m].mean())
        print(f"{c:15s} {m.sum():6d} {means[-1]:15.3f} {rates[-1]:20.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["loss"]); ax[0].set(title="MIL training loss", xlabel="Epoch")
    ax[1].plot(hist["val_auc"]); ax[1].set(title="Validation video-level AUC", xlabel="Epoch")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "training_curves.png"), dpi=150); plt.close()

    fpr, tpr, _ = roc_curve(yte, vs)
    plt.figure(figsize=(5.5, 5))
    plt.plot(fpr, tpr, label=f"Video-level AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC - UCF-Crime test videos"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "roc_curve.png"), dpi=150); plt.close()

    plt.figure(figsize=(10, 4.5))
    cols = ["tab:blue" if not is_anom(c) else "tab:red" for c in classes]
    plt.bar(range(len(classes)), rates, color=cols)
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.ylabel("Fraction flagged as anomalous")
    plt.title("Detection rate per class (blue = normal, i.e. false-alarm rate)")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "class_detection.png"), dpi=150); plt.close()

    an = [i for i in range(len(te_keep)) if yte[i]][:4]
    no = [i for i in range(len(te_keep)) if not yte[i]][:2]
    sel = an + no
    if sel:
        fig, axs = plt.subplots(len(sel), 1, figsize=(10, 2.0 * len(sel)), squeeze=False)
        for ax_, i in zip(axs[:, 0], sel):
            ax_.plot(S[i], color="tab:red" if yte[i] else "tab:blue")
            ax_.set(ylim=(0, 1), ylabel="score", title=f"{te_keep[i]['cls']} / {te_keep[i]['vid']}")
        axs[-1, 0].set_xlabel("Segment (video time ->)")
        plt.tight_layout(); plt.savefig(os.path.join(args.out, "score_timelines.png"), dpi=150); plt.close()
    print(f"\nSaved model + graphs to ./{args.out}/")


def main():
    args = get_args()
    root = args.root
    tr_dir, te_dir = find_split_dir(root, "train"), find_split_dir(root, "test")
    if not tr_dir or not te_dir:
        raise SystemExit(f"Could not find Train/ and Test/ folders under {root}")
    print("Indexing frames (can take a minute)...")
    train_all, test_all = index_items(tr_dir, "train"), index_items(te_dir, "test")
    auto = bool(train_all) and needs_chunking(train_all)
    if args.stage == "inspect":
        inspect(train_all, test_all, root)
        if auto:
            print("\nNOTE: about one 'video' per class was found, so the frames look combined per class. "
                  "The run will cut each class into pseudo-videos of 640 consecutive frames "
                  "(change with --chunk_frames).")
        return
    if not train_all or not test_all:
        inspect(train_all, test_all, root)
        raise SystemExit("No frames could be parsed.")

    chunk = args.chunk_frames if args.chunk_frames >= 0 else (640 if auto else 0)
    if chunk > 0:
        train_all = rechunk(train_all, chunk, args.clip_len)
        test_all = rechunk(test_all, chunk, args.clip_len)
        print(f"Splitting frames into pseudo-videos of {chunk} frames: "
              f"{len(train_all)} train, {len(test_all)} test")

    rng = np.random.default_rng(args.seed)
    train_items = subsample(train_all, args.train_per_class, rng, args.prefer_small)
    test_items = subsample(test_all, args.test_per_class, rng, args.prefer_small)
    print(f"Using {len(train_items)} train + {len(test_items)} test videos")
    os.makedirs(args.out, exist_ok=True)
    for tag, lst in (("train", train_items), ("test", test_items)):
        with open(os.path.join(args.out, f"subset_{tag}.txt"), "w") as f:
            f.write("\n".join(it["name"] for it in lst))

    if args.stage in ("extract", "all"):
        extract(args, train_items + test_items)
    if args.stage in ("train", "all"):
        train_and_eval(args, train_items, test_items)


if __name__ == "__main__":
    main()