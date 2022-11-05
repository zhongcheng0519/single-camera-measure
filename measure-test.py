#!/usr/bin/env python
from Calib.CameraCalibration import CameraCalibration
import cv2


# TEST_IMG = 'train/20220126131357.jpg'
TEST_IMG = 'train2/IMG_20220617_162053.jpg'


def main():
    # Extracting path of individual image stored in a given directory
    cam_calib = CameraCalibration()
    cam_calib.verbose = True
    cam_calib.load_param()

    # 先找外参
    img = cv2.imread(TEST_IMG, cv2.IMREAD_GRAYSCALE)
    imgpoints, objpoints, rvec, tvec = cam_calib.getExteriorParameter(img)
    print('imgpoints=', imgpoints)
    print('objpoints=', objpoints)
    print('rvec=', rvec)
    print('tvec=', tvec)

    # 再去畸变
    undistort_img = cam_calib.undistort(img)

    # 再重新找imgpoints
    imgpoints = cam_calib.findBoardCorners(undistort_img)

    # 再重新反投影
    Pw = cam_calib.backProjectPoints(imgpoints, rvec, tvec)
    print('Points in world coordinate:')
    print(Pw)

    # Compare Pw and objpoints
    error = cv2.norm(Pw, objpoints, cv2.NORM_L2) / len(Pw)
    print("total back project error: {}".format(error))


if __name__ == "__main__":
    main()
