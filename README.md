# Video Anomaly Detection on UCF-Crime

Weakly-supervised video anomaly detection using a **frozen Kinetics-pretrained 3D CNN (R3D-18)**, a **BiGRU** temporal head and a **Multiple-Instance Learning (MIL) ranking loss**. Trained and evaluated on the frame-extracted [UCF-Crime dataset from Kaggle](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset) with only **video-level labels** (no temporal annotations).

**Result: video-level test AUC = 0.844** on the 290 official UCF-Crime test videos.

![ROC curve]([results/roc_curve.png](https://github.com/Khurram32/ucf-crime-anomaly-detection-BiGRU-based-3D-CNN/blob/Khurram32-results/roc_curve.png))

## How it works

```
frames (PNG)  ->  16-frame clips  ->  R3D-18 (frozen, Kinetics-400)  ->  512-d clip features
      ->  mean-pool into T = 32 segments, L2-normalise
      ->  Linear(512,256) + ReLU + Dropout  ->  BiGRU(256 -> 2x128)
      ->  MLP head + Sigmoid  ->  anomaly score per segment  (video score = max over segments)
```

- **Weak supervision (MIL).** Each video is a "bag" of 32 segments. Only the video label is known (`NormalVideos` = normal, every other class = anomaly). The loss is a ranking hinge, `max(0, 1 - max(score_anomalous) + max(score_normal))`, plus temporal-smoothness and sparsity terms (both 8e-5).
- **Only the head is trained** (444,289 parameters); the 3D CNN is frozen and features are cached, so training takes minutes once features exist.
- **Threshold selection** uses Youden's J on a held-out validation split (10% of training videos, stratified), then is applied unchanged to the test set. The checkpoint with the best validation AUC is kept.
- **Training details:** AdamW (lr 1e-3, weight decay 1e-3), OneCycleLR, batch of 30 anomalous + 30 normal bags, 100 epochs, Gaussian feature noise (σ = 0.01), gradient clipping at 5.0, seed 42.

## Results

| Metric (test, video-level) | Value |
|---|---|
| ROC AUC | **0.844** |
| False-alarm rate on normal videos | ~9% (see class chart) |

Running the script prints the full metric set (AP, accuracy, precision, recall, per-class table).

| Per-class detection rate | Training curves |
|---|---|
| ![Class detection]([results/class_detection.png](https://github.com/Khurram32/ucf-crime-anomaly-detection-BiGRU-based-3D-CNN/blob/Khurram32-results/class_detection.png)) | ![Training curves]([results/training_curves.png](https://github.com/Khurram32/ucf-crime-anomaly-detection-BiGRU-based-3D-CNN/blob/Khurram32-results/training_curves.png)) |

**Segment-level score timelines** for sample test videos (red = anomalous, blue = normal):

![Score timelines]([results/score_timelines.png](https://github.com/Khurram32/ucf-crime-anomaly-detection-BiGRU-based-3D-CNN/blob/Khurram32-results/score_timelines.png))

### Observations

- Strongest detection on **Burglary, Stealing, Vandalism and Explosion** (roughly 75-85% flagged); weakest on **Shoplifting and Robbery** (roughly 33-40%). Subtle, short or low-motion events are harder for a clip-level motion feature.
- Validation AUC is noisy (it dips sharply around epoch 27 and again near 75) because the validation set is small (about 160 videos) and the MIL objective is unstable; best-checkpoint selection mitigates this.
- Some anomalous videos are missed entirely (e.g. `Abuse028` stays at 0 across all segments), and some score near 1 from the very first segment (e.g. `Arrest007`).

## Limitations

- **Video-level evaluation only.** The Kaggle frame dataset has no temporal annotations, so frame-level AUC (the metric used in most UCF-Crime papers) cannot be computed. Numbers here are **not directly comparable** to published frame-level AUCs.
- Frames in the Kaggle release are low-resolution and sparsely sampled (resized to 112x112 here).
- The validation split is drawn from the training set; there is no separate held-out tuning set beyond that.
- Single seed, single run - no confidence intervals.

## Repository layout

```
.
├── ucf_crime_anomaly_frames.py   # full pipeline: indexing, feature extraction, training, evaluation, plots
├── checkpoints/best_model.pt     # trained BiGRU head (state dict + T)
├── splits/                       # exact train / test video lists used
├── results/                      # ROC, training curves, per-class detection, score timelines
├── requirements.txt
├── LICENSE
└── CITATION.cff
```

## Getting started

```bash
git clone https://github.com/Khurram32/ucf-crime-anomaly-detection.git
cd ucf-crime-anomaly-detection
pip install -r requirements.txt
```

### 1. Get the data

Download [odins0n/ucf-crime-dataset](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset) from Kaggle and unzip it so it contains `Train/` and `Test/` folders:

```
Train/<ClassName>/<video>_<frameNumber>.png
Test/<ClassName>/<video>_<frameNumber>.png
```

### 2. Run

```bash
python ucf_crime_anomaly_frames.py --root /path/to/dataset --stage inspect   # check structure
python ucf_crime_anomaly_frames.py --root /path/to/dataset --stage all       # extract + train + evaluate
```

Useful flags: `--stage {inspect,extract,train,all}`, `--epochs`, `--T`, `--clip_len`, `--max_clips`, `--train_per_class` / `--test_per_class` (quick subset runs), `--seed`. Outputs are written to `outputs_frames/`; features are cached in `features_frames/`. A GPU is recommended for feature extraction.

### 3. Use the pretrained head

```python
import torch
from ucf_crime_anomaly_frames import GRUAnomaly, to_bag

ckpt = torch.load("checkpoints/best_model.pt", map_location="cpu")
model = GRUAnomaly()
model.load_state_dict(ckpt["model"])
model.eval()

# clip_feats: (n_clips, 512) R3D-18 features of one video
bag = torch.from_numpy(to_bag(clip_feats, ckpt["T"])).unsqueeze(0)   # (1, 32, 512)
with torch.no_grad():
    segment_scores = model(bag)[0]          # (32,) values in [0, 1]
video_score = segment_scores.max().item()   # higher = more anomalous
```

Features must come from the same extractor (torchvision `r3d_18`, Kinetics weights, 112x112 input, 16-frame clips, Kinetics mean/std) - see `extract()` in the script.

## Dataset and acknowledgements

- Sultani, W., Chen, C., Shah, M. *Real-world Anomaly Detection in Surveillance Videos.* CVPR 2018 (UCF-Crime).
- Frame-extracted version: [odins0n/ucf-crime-dataset](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset) on Kaggle. Please follow the dataset's own terms of use; the data is **not** redistributed in this repository.
- Backbone: torchvision `r3d_18` pretrained on Kinetics-400.

## Author

**Khurram** - [github.com/Khurram32](https://github.com/Khurram32)
**Sahil** -(https://github.com/SahilRai02)
**TejasA03** 

Released under the [MIT License](LICENSE).
