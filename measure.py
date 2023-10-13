import numpy as np
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
from mpl_interactions import zoom_factory, panhandler
import cv2
from Calib.CameraCalibration import CameraCalibration


TEST_IMG = 'test/02.png'


def main():
    # Extracting path of individual image stored in a given directory
    cam_calib = CameraCalibration()
    cam_calib.verbose = True
    cam_calib.load_param()

    # test exterior parameter first
    img = cv2.imread(TEST_IMG, cv2.IMREAD_GRAYSCALE)
    _, _, rvec, tvec = cam_calib.getExteriorParameter(img)
    print('rvec=', rvec)
    print('tvec=', tvec)

    # undistort second
    undistort_img = cam_calib.undistort(img)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.imshow(undistort_img, cmap="gray")

    # add zooming and middle click to pan
    zoom_factory(ax)
    ph = panhandler(fig, button=2)

    klicker = clicker(
       ax,
       ["event"],
       markers=["o"],
       colors=['r'],
       linestyle="-"
    )
    plt.show()

    imgpoints = klicker.get_positions()['event']
    # imgpoints[:, [1, 0]] = imgpoints[:, [0, 1]]
    print("imgpoints=", imgpoints)

    Pw = cam_calib.backProjectPoints(imgpoints, rvec, tvec)
    print('Points in world coordinate:', Pw)

    dists = list()
    for i in range(len(Pw)-1):
        diff = Pw[i+1]-Pw[i]
        dists.append(np.linalg.norm(diff))

    print("距离: ")
    print(dists)


if __name__ == "__main__":
    main()
