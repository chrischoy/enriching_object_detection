# Enriching Object Detection

This is a code repository for my CVPR 15 paper, Enriching Object Detection with
2D-3D registration and continuous viewpoint estimation.

## Dependencies

- OSGRenderer [https://github.com/chrischoy/OSGRenderer]

Put the OSGRenderer in the parent directory


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


```
git clone https://github.com/chrischoy/EnrichingObjectDetection
git clone https://github.com/chrischoy/OSGRenderer
cd OSGRenderer
matlab -r compile
cd ../EnrichingObjectDetection
matlab -r 'eod_compile'
```
## Issues

### Mac OS Matlab can't find a compiler

Follow [the instruction](http://www.mathworks.com/matlabcentral/answers/246507#answer_194526)
and setup the xcode compiler

