"""Object detection using SSD (Single Shot MultiBox Detector).

Processes a video file frame-by-frame, runs SSD inference on each frame,
draws bounding boxes and class labels, and writes the annotated video.
"""

import os
import torch
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio


def detect(frame, net, transform):
    """Run SSD detection on a single frame and draw bounding boxes.

    Args:
        frame: Input image as a numpy array (H, W, C).
        net: The SSD network in eval mode.
        transform: BaseTransform instance for preprocessing.

    Returns:
        Annotated frame with bounding boxes and labels drawn.
    """
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    x = x.unsqueeze(0)

    with torch.no_grad():
        y = net(x)

    detections = y.data
    scale = torch.Tensor([width, height, width, height])

    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])),
                          (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
                        cv2.LINE_AA)
            j += 1
    return frame


def main():
    """Load SSD model, process input video, and write annotated output."""
    net = build_ssd('test')
    net.load_state_dict(
        torch.load('ssd300_mAP_77.43_v2.pth',
                    map_location=lambda storage, loc: storage,
                    weights_only=True)
    )
    net.eval()

    transform = BaseTransform(net.size, (104 / 256.0, 117 / 256.0, 123 / 256.0))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_video = os.path.join(script_dir, 'man-and-dog.mp4')
    output_video = os.path.join(script_dir, 'output.mp4')

    reader = imageio.get_reader(input_video)
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(output_video, fps=fps)

    for i, frame in enumerate(reader):
        frame = detect(frame, net, transform)
        writer.append_data(frame)
        print(f"Processed frame {i}")

    writer.close()
    print("Done! Output saved to", output_video)


if __name__ == "__main__":
    main()
