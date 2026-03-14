# 🎯 Object Detection using Single Shot Detection (SSD)

Real-time object detection on video using the **SSD300** (Single Shot MultiBox Detector) architecture with a VGG-16 backbone. The model detects and classifies 20 object categories from the PASCAL VOC dataset in a single forward pass — no region proposal step needed.

> 📄 Based on the paper: [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg (2016)

---

## 🏗️ Architecture

```
Input Image (300×300)
       │
   ┌───▼───┐
   │ VGG-16 │  (pretrained base network, FC layers removed)
   │ conv1  │──► conv4_3 ──► L2Norm ──► predictions (4 boxes/cell)
   │  ...   │
   │ conv5  │──► fc7 (as conv) ──────► predictions (6 boxes/cell)
   └───┬───┘
       │
   ┌───▼────────┐
   │ Extra Convs │  (auxiliary feature extraction layers)
   │ conv8_2     │──► predictions (6 boxes/cell)
   │ conv9_2     │──► predictions (6 boxes/cell)
   │ conv10_2    │──► predictions (4 boxes/cell)
   │ conv11_2    │──► predictions (4 boxes/cell)
   └─────────────┘
       │
   ┌───▼───┐
   │  NMS  │  Non-Maximum Suppression
   └───┬───┘
       │
   Final Detections (class + bounding box + confidence)
```

Multi-scale feature maps (38×38, 19×19, 10×10, 5×5, 3×3, 1×1) allow the network to detect objects at different sizes. Each feature map cell predicts offsets for a set of default (prior) boxes and per-class confidence scores.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| 🧠 Deep Learning | PyTorch |
| 👁️ Computer Vision | OpenCV |
| 🎬 Video I/O | imageio |
| 📊 Data Format | PASCAL VOC (XML annotations) |
| 🏛️ Base Network | VGG-16 (pretrained on ImageNet) |

---

## 📦 Dependencies

```
torch >= 1.0
torchvision
opencv-python
imageio
imageio-ffmpeg
numpy
Pillow
```

Install all dependencies:

```bash
pip install torch torchvision opencv-python imageio imageio-ffmpeg numpy Pillow
```

---

## 🚀 How to Run

### 1. Download pretrained weights

Download `ssd300_mAP_77.43_v2.pth` from the [original SSD PyTorch repo](https://github.com/amdegroot/ssd.pytorch) and place it in the `object_detection with SSD/` directory.

### 2. Run object detection on a video

```bash
cd "object_detection with SSD"
python object_detection.py
```

This will:
- Load the SSD300 model with pretrained VOC weights
- Process `man-and-dog.mp4` frame by frame
- Draw bounding boxes and class labels on detected objects
- Write the result to `output.mp4`

To use a different video, edit the filename in `object_detection.py`:
```python
reader = imageio.get_reader('your-video.mp4')
```

### 3. (Optional) Train on PASCAL VOC

Download the VOC dataset using the provided scripts:
```bash
cd "object_detection with SSD /data/scripts"
bash VOC2007.sh
bash VOC2012.sh
```

---

## 📁 Project Structure

```
├── object_detection with SSD/
│   ├── ssd.py                  # SSD network definition + build_ssd()
│   ├── object_detection.py     # Video inference pipeline
│   ├── man-and-dog.mp4         # Sample input video
│   └── output.mp4              # Generated output
│
├── object_detection with SSD /
│   ├── data/
│   │   ├── __init__.py         # BaseTransform + data utilities
│   │   ├── config.py           # SSD300 v1/v2 hyperparameters
│   │   └── voc0712.py          # VOC dataset loader
│   └── layers/
│       ├── box_utils.py        # NMS, IoU, encode/decode utilities
│       ├── functions/
│       │   ├── detection.py    # Post-processing (decode + NMS)
│       │   └── prior_box.py    # Default/prior box generation
│       └── modules/
│           ├── l2norm.py       # L2 normalization layer
│           └── multibox_loss.py # SSD training loss (loc + conf)
│
├── LICENSE
└── README.md
```

---

## ⚠️ Known Issues

- Only **SSD300** is implemented; SSD512 config stubs exist but are empty
- The pretrained weights file (`ssd300_mAP_77.43_v2.pth`) is not included in the repo — must be downloaded separately
- Detection confidence threshold is hardcoded at `0.6` in `object_detection.py`
- No GPU/CUDA inference path in the video detection script (runs on CPU by default)
- The repo contains two similarly-named directories (`object_detection with SSD` and `object_detection with SSD ` with a trailing space) — this is a legacy structure from the original repo

---

## 🙏 Credits

- SSD PyTorch implementation by [Max de Groot (amdegroot)](https://github.com/amdegroot/ssd.pytorch)
- Original paper by Wei Liu et al. — [arXiv:1512.02325](https://arxiv.org/abs/1512.02325)
- VGG-16 architecture by Karen Simonyan & Andrew Zisserman

---

## 📄 License

See [LICENSE](LICENSE) for details.
