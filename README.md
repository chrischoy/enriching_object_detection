# Enriching Object Detection

This is a code repository for my CVPR 15 paper, Enriching Object Detection with
2D-3D registration and continuous viewpoint estimation.

- [Project Page](http://chrischoy.github.io/blog/projects/enrich-object-detection/)
- [Paper](http://chrischoy.github.io/blog/external/cvpr15/choy_cvpr15.pdf)
- [Supplementary Paper](http://chrischoy.github.io/blog/external/cvpr15/choy_cvpr15_supp.pdf)

```
@InProceedings{choy_cvpr15,
  author       = {Christopher B. Choy and Michael Stark and Sam Corbett-Davies and Silvio Savarese},
  title        = {Enriching Object Detection with 2D-3D Registration and Continuous Viewpoint Estimation},
  booktitle    = {Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition},
  year         = {2015},
}
```

## Dependencies

- OSGRenderer [https://github.com/chrischoy/OSGRenderer]

There are two modes of operation. EOD works as a standalone detector and also
as an detection augmentation pipeline. The demo code contains both versions of
mode.

## Installation

- install OpenSceneGraph
- Install the latest CUDA [link](https://developer.nvidia.com/cuda-downloads)
- Install [OSGRenderer](https://github.com/chrischoy/OSGRenderer)

The OSGRenderer requires OpenSceneGraph

To install on Mac OS X, type

```
brew install open-scene-graph
```

To install on Linux, type

```
sudo apt-get install openscenegraph
```

Next, install the OSG and this code repository and install by typing

```
git clone https://github.com/chrischoy/EnrichingObjectDetection
git clone https://github.com/chrischoy/OSGRenderer
cd OSGRenderer
matlab -r compile
cd ../EnrichingObjectDetection
matlab -r 'eod_compile'
```

## Demo

First, the demo code will generate NZ-WHO detectors. The detectors are generated on the fly without training.

![NZ-WHO detectors](https://raw.githubusercontent.com/chrischoy/enriching_object_detection/master/data/readme/detectors.gif)

Next, using the detectors, we can detect objects in the image. However, we can feed in the detection results from any other off-the-shelf detectors.

![Detections](https://raw.githubusercontent.com/chrischoy/enriching_object_detection/master/data/readme/detections.gif)

Using our detectors, we can get the detection results.

![detection result](https://raw.githubusercontent.com/chrischoy/enriching_object_detection/master/data/readme/detection_result.png)

Finally, we finely tune the detection result using MCMC stage.

![Detections](https://raw.githubusercontent.com/chrischoy/enriching_object_detection/master/data/readme/mcmc.gif)

After the tuning,

![tuning result 1](https://raw.githubusercontent.com/chrischoy/enriching_object_detection/master/data/readme/tuning_result_1.png)
![tuning result 2](https://raw.githubusercontent.com/chrischoy/enriching_object_detection/master/data/readme/tuning_result_2.png)
![tuning result 3](https://raw.githubusercontent.com/chrischoy/enriching_object_detection/master/data/readme/tuning_result_3.png)
![tuning result 4](https://raw.githubusercontent.com/chrischoy/enriching_object_detection/master/data/readme/tuning_result_4.png)

## Issues

### Mac OS Matlab can't find a compiler

Follow [the instruction](http://www.mathworks.com/matlabcentral/answers/246507#answer_194526)
and setup the xcode compiler

