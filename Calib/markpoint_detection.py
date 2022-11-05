import cv2
import numpy as np


def detect_mark_point(img, diameter):
    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 80
    params.maxThreshold = 120

    # Filter by Area.
    params.filterByArea = True
    area = (diameter/2)**2*np.pi
    params.minArea = area*0.5
    params.maxArea = area*1.5

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.8

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.8

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.8

    params.filterByColor = False

    detector = cv2.SimpleBlobDetector_create(params)
    # Detect blobs.
    keypoints = detector.detect(img)

    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # cv2.imshow("Keypoints", im_with_keypoints)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if len(keypoints) != 1:
        return None, None
    else:
        return keypoints[0].pt, keypoints[0].size


if __name__ == '__main__':
    image = cv2.imread('/home/xingzhi/MyCode/5Axis/5Axis_Inovance3/data/tofindmark.png')

    pt, sz = detect_mark_point(image, 24)
    print('center = ', pt)
    print('size = ', sz)
