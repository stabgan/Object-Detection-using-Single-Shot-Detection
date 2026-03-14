# Object Detection using Single Shot Detection (SSD)

Real-time object detection on video using the SSD300 architecture with a VGG-16 backbone, pretrained on Pascal VOC.

## What It Does

Processes a video file frame-by-frame through an SSD300 network, detects objects from 20 Pascal VOC classes, draws bounding boxes with class labels, and writes the annotated result to a new video file.

## Architecture

SSD (Single Shot MultiBox Detector) performs object localization and classification in a single forward pass:

- **Base network**: VGG-16 (truncated before fully connected layers)
- **Feature extraction**: Multi-scale feature maps from auxiliary convolutional layers (conv6 onwards)
- **Detection heads**: Separate conv layers for bounding box regression and class confidence at each scale
- **Post-processing**: Non-maximum suppression (NMS) to filter overlapping detections

The model scores **74%+ mAP** at 59 FPS on Pascal VOC.

> Paper: [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) (Liu et al., 2016)

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| 🧠 Model | SSD300 (VGG-16 backbone) |
| 🔥 Framework | PyTorch |
| 📹 Video I/O | imageio, OpenCV |
| 📦 Dataset | Pascal VOC 2007/2012 (20 classes) |

## Setup

```bash
# Install dependencies
pip install torch torchvision opencv-python imageio imageio-ffmpeg numpy

# Download pretrained weights (~100 MB)
# Place ssd300_mAP_77.43_v2.pth in the project directory
```

## Usage

```bash
cd "object_detection with SSD"
python object_detection.py
```

Edit `object_detection.py` to change the input video file path. By default it processes `man-and-dog.mp4`.

## Project Structure

```
object_detection with SSD/     # Main scripts
├── object_detection.py        # Video inference pipeline
├── ssd.py                     # SSD model definition
├── man-and-dog.mp4            # Sample input video
└── epic_horses.mp4            # Additional sample video

object_detection with SSD /    # Supporting modules
├── data/                      # Dataset classes and config
│   ├── config.py              # SSD300/512 hyperparameters
│   └── voc0712.py             # VOC dataset loader
└── layers/                    # Network components
    ├── box_utils.py           # Bounding box utilities (NMS, IoU, encode/decode)
    ├── functions/
    │   ├── detection.py       # Post-processing detection layer
    │   └── prior_box.py       # Default anchor box generation
    └── modules/
        ├── l2norm.py          # L2 normalization layer
        └── multibox_loss.py   # SSD training loss (smooth L1 + cross entropy)
```

## Sample Output

![Detection example](https://image.ibb.co/dJraGS/image.png)

[Full output video on YouTube](https://youtu.be/OrhB3qGQhZI)

## ⚠️ Known Issues

- The pretrained weights file (`ssd300_mAP_77.43_v2.pth`) is not included in the repo — must be downloaded separately
- Two directories with near-identical names exist due to a trailing space; both are required for imports to resolve correctly
- Only SSD300 is implemented; SSD512 config entries are empty placeholders

## Credits

SSD PyTorch implementation based on [amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) by Max de Groot.
