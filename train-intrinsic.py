#!/usr/bin/env python
import glob
from Calib.CameraCalibration import CameraCalibration


def main():
    # Extracting path of individual image stored in a given directory
    img_names = glob.glob('./train/*.png')
    cam_calib = CameraCalibration()
    cam_calib.load_param()
    cam_calib.verbose = True
    cam_calib.getInteriorParameter(img_names)
    cam_calib.save_param()


if __name__ == "__main__":
    main()
