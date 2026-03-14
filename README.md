# Object Detection

[output video](https://youtu.be/OrhB3qGQhZI)

### The paper about SSD: Single Shot MultiBox Detector (by C. Szegedy et al.) was released at the end of November 2016 and reached new records in terms of performance and precision for object detection tasks, scoring over 74% mAP (mean Average Precision) at 59 frames per second on standard datasets such as PascalVOC and COCO. The name of this architecture comes from:

- Single Shot: this means that the tasks of object localization and classification are done in a single forward pass of the network
- MultiBox: this is the name of a technique for bounding box regression developed by Szegedy et al. (we will briefly cover it shortly)
- Detector: The network is an object detector that also classifies those detected objects

# Architecture :
![](https://cdn-images-1.medium.com/max/1600/1*51joMGlhxvftTxGtA4lA7Q.png)

SSD’s architecture builds on the venerable VGG-16 architecture, but discards the fully connected layers. The reason VGG-16 was used as the base network is because of its strong performance in high quality image classification tasks and its popularity for problems where transfer learning helps in improving results. Instead of the original VGG fully connected layers, a set of auxiliary convolutional layers (from conv6 onwards) were added, thus enabling to extract features at multiple scales and progressively decrease the size of the input to each subsequent layer.

![](https://cdn-images-1.medium.com/max/1600/1*3-TqqkRQ4rWLOMX-gvkYwA.png)

#### Multibox :
The bounding box regression technique of SSD is inspired by Szegedy’s work on MultiBox, a method for fast class-agnostic bounding box coordinate proposals. Interestingly, in the work done on MultiBox an Inception-style convolutional network is used. The 1x1 convolutions that you see below help in dimensionality reduction since the number of dimensions will go down (but “width” and “height” will remain the same).

![](https://cdn-images-1.medium.com/max/1600/1*WbNf0ngkmCJYT_jXX6IaOw.png)

# Thanks to [Max de Groot](https://github.com/amdegroot/ssd.pytorch) for the ssd.py

# Now the fun part ! RESULTS THAT I OBTAINED ARE NOT 100% ACCURATE BUT IT's GOOD :

After running the man-and-dog.mp4 file i got  :

Output :

![](https://image.ibb.co/b1ZCwS/frame_050_delay_0_1s.gif)
![](https://image.ibb.co/dJraGS/image.png)
![](https://image.ibb.co/jR8pbS/frame_069_delay_0_1s.gif)
