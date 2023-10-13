# 单目视觉测量

*[英文](README.md)*

## 简介

现在的手机照相功能使得照片随手可得。大多数人不会使用相机来进行测量。但是这在工程上实际上是非常实用的方法。这里介绍的单目视觉测量适用于用手机来对某个特定平面进行测量。

这个项目基于`OpenCV`库，使用三个步骤来完成来实现单目测量方法：

1. 用棋盘格进行内参校准；
2. 用棋盘格获取外参；（根据实际场景，也可以选取不同的方法来进行外参标定。）
3. 在步骤2拍摄的图片上，选取线段。程序会显示线段对应的尺寸。

## 使用方法

### 步骤一、训练内参

1. 用相机从不同的角度拍摄棋盘格，将图片放于`train`文件夹中。
2. 修改`data/Calibration.toml`中的`chess_size`、`grid_length`以及`image_size`。其中，`chess_size`为棋盘格的格子数，`grid_length`为每一正方形格子的尺寸，单位是`mm`。`image_size`指的是图像的大小。
3. 调用`train-intrinsic.py`。得到的内参结果会保存在`data/Calibration.toml`中。

### 步骤二、标定外参



### 步骤三、测量
