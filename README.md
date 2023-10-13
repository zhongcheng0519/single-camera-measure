# Single Camera Measure

*Read this in other languages: [中文](README_zh.md).*

## Introduction

This project is used for length measurement by a single camera, based on `OpenCV`.

It is done by the well-known three steps:

1. Calibrate the intrinsic parameters of a camera by a set of chessboard images.
2. Get Extrinsic parameters in the specified pose according to your application. Chessboard image is also needed in this step.
3. Click and drag a line in the image within the same plane of the chessboard in step 2, and convert the pixel length to real world length.

## Usage

Step 1: Train intrinsic parameters

Step 2: Obtain extrinsic parameters

Step 3: Measure

