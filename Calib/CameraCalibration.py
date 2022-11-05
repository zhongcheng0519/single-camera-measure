import os.path
import numpy as np
import cv2
from Calib.markpoint_detection import detect_mark_point
import toml


class CameraCalibration(object):
    def __init__(self):
        self.verbose = False
        self.param_path = './data/Calibration.toml'
        self.new_mtx = None
        self.new_mapx = None
        self.new_mapy = None
        self.new_center = None
        self.params = dict()

    def load_param(self):
        """
        load param
        """
        self.params = toml.load(self.param_path)
        print(self.params)
        if "mtx" in self.params:
            self.params["mtx"] = np.array(self.params["mtx"])
        if "dist" in self.params:
            self.params["dist"] = np.array(self.params["dist"])
        if "rvec" in self.params:
            self.params["rvec"] = np.array(self.params["rvec"])
        if "tvec" in self.params:
            self.params["tvec"] = np.array(self.params["tvec"])

    def save_param(self):
        """
        save param
        """
        if "mtx" in self.params:
            self.params["mtx"] = self.params["mtx"].tolist()
        if "dist" in self.params:
            self.params["dist"] = self.params["dist"].tolist()
        if "rvec" in self.params:
            self.params["rvec"] = self.params["rvec"].tolist()
        if "tvec" in self.params:
            self.params["tvec"] = self.params["tvec"].tolist()
        with open(self.param_path, "w") as toml_file:
            toml.dump(self.params, toml_file)

    def gen_obj_img_points(self, image_names, objp):
        """
        generate object points and image points
        :param image_names: image names
        :param objp: object points
        :return: imgpoints, objpoints
        """
        patternsize = (self.params['chess_size'][1], self.params['chess_size'][0])
        # Arrays to store object points and image points from all the images.
        objpoints = []
        imgpoints = []
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        for fname in image_names:
            print(fname)
            img = cv2.imread(fname, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, patternsize, None)

            # If found, add object points, image points (after refining them)
            if ret is True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                if self.verbose:
                    filename = os.path.join('draw-corners', os.path.basename(fname))
                    cv2.imwrite(filename, cv2.drawChessboardCorners(img, patternsize, corners2, ret))
                    # cv2.imshow('img', img)
                    # cv2.setWindowTitle('img', fname)
                    # cv2.waitKey(0)
            else:
                print("findChessboardCorners failed!")

        # if self.verbose:
        #     cv2.destroyAllWindows()

        return imgpoints, objpoints

    def undistort(self, img, mtx=None):
        if isinstance(img, str):
            output_name = f'undistort-img/{os.path.basename(img)}'
            img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        else:
            output_name = 'undistort-img/undistort.jpg'
        mtx = self.params['mtx']
        dist = self.params['dist']
        h, w = img.shape[:2]
        # if new_mtx is None or roi is None:
        #     new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (w, h), 5)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        # crop the image
        # if need_crop:
        #     x, y, w, h = roi
        #     dst = dst[y:y + h, x:x + w]

        if self.verbose:
            cv2.imwrite(output_name, dst)
            # cv2.imshow('undistort', dst)
            # cv2.waitKey(0)
        return dst

    def undistort_rotate(self, img, roi=None, need_crop=False, rvec=None):
        if self.params['mtx'] is None or self.params['dist'] is None or self.params['rvec'] is None:
            print('error read params')
            return

        h, w = img.shape[:2]
        if self.new_mtx is None:
            self.new_mtx, roi = cv2.getOptimalNewCameraMatrix(self.params['mtx'], self.params['dist'], (w, h), 1,
                                                              (w, h))

        if self.new_mapx is None or self.new_mapy is None or self.new_center is None:
            mapx, mapy = cv2.initUndistortRectifyMap(self.params['mtx'], self.params['dist'], None, self.new_mtx,
                                                     (w, h), 5)
            R, _ = cv2.Rodrigues(np.array(self.params['rvec']))
            self.new_mapx = R[0][0] * mapx + R[0][1] * mapy
            self.new_mapy = R[1][0] * mapx + R[1][1] * mapy
            self.new_center = self.map_pt(self.new_mapx, self.new_mapy, [(w - 1) / 2, (h - 1) / 2])
        dst = cv2.remap(img, self.new_mapx, self.new_mapy, cv2.INTER_LINEAR)
        # crop the image
        if need_crop:
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]

        if self.verbose:
            cv2.imwrite('undistort-img/undistort_rotate.jpg', dst)
            # cv2.imshow('undistort', dst)
            # cv2.waitKey(0)
        return dst, self.new_center

    def map_pt(self, mapx, mapy, pt):
        ret_pt = [int(pt[0]), int(pt[1])]
        while True:
            if mapx[ret_pt[1], ret_pt[0]] + 1 < pt[0]:
                ret_pt[0] += 1
            elif mapx[ret_pt[1], ret_pt[0]] > pt[0]:
                ret_pt[0] -= 1
            elif mapy[ret_pt[1], ret_pt[0]] + 1 < pt[1]:
                ret_pt[1] += 1
            elif mapy[ret_pt[1], ret_pt[0]] > pt[1]:
                ret_pt[1] -= 1
            else:
                break
        x_prop = pt[0] - mapx[ret_pt[1], ret_pt[0]]
        y_prop = pt[1] - mapy[ret_pt[1], ret_pt[0]]
        newx = (1 - x_prop) * ret_pt[0] + x_prop * (ret_pt[0] + 1)
        newy = (1 - y_prop) * ret_pt[1] + y_prop * (ret_pt[1] + 1)
        # newx_down = (1 - x_prop)*ret_pt[0] + x_prop * mapx[ret_pt[0] + 1, ret_pt[1] + 1]
        # newx = (1-y_prop)*newx_up + y_prop*newx_down
        #
        # newy_up = (1 - x_prop) * mapy[tuple(ret_pt)] + x_prop * mapy[ret_pt[0] + 1, ret_pt[1]]
        # newy_down = (1 - x_prop) * mapy[ret_pt[0], ret_pt[1] + 1] + x_prop * mapy[ret_pt[0] + 1, ret_pt[1] + 1]
        # newy = (1 - y_prop) * newy_up + y_prop * newy_down
        return [newx, newy]

    def backProjectPoints(self, image_points, rvec, tvec) -> np.ndarray:
        """
        project image points to object points
        """
        mtx = self.params['mtx']
        pi = image_points.reshape((-1, 2))
        uv1 = np.hstack([pi, np.ones((pi.shape[0], 1))])
        R, _ = cv2.Rodrigues(rvec)
        rotated_Pc = np.linalg.inv(mtx @ R) @ uv1.T
        rotated_t = np.linalg.inv(R) @ tvec
        Zc = rotated_t[2] / rotated_Pc[2, :]
        Pw = rotated_Pc * Zc - rotated_t
        return np.transpose(Pw).astype(np.float32)

    def getInteriorParameter(self, img_names):
        """
        Get Camera's Interior Parameter
        :param img_names: list of image names
        :return: Interior Parameter
        """
        objp = np.zeros((self.params['chess_size'][1] * self.params['chess_size'][0], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.params['chess_size'][1], 0:self.params['chess_size'][0]].T.reshape(-1, 2)
        objp = objp * self.params['grid_length']

        imgpoints, objpoints = self.gen_obj_img_points(img_names, objp)
        w, h = cv2.imread(img_names[0], cv2.IMREAD_GRAYSCALE).shape[::-1]
        ret, self.params['mtx'], self.params['dist'], rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

        if self.verbose:
            for img_name in img_names:
                self.undistort(img_name)

        if not ret:
            print("Calibration Failed!")
            return None, None
        self.params['image_size'] = [w, h]

        if self.verbose is True:
            print("\nRMS:", ret)
            print("camera matrix:\n", self.params['mtx'])
            print("distortion coefficients: ", self.params['dist'].ravel())
            print("rvecs:\n", rvecs)

        # Re-project to see error
        mean_error = 0
        print("verify object points to image points")
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], self.params['mtx'], self.params['dist'])
            imgpoints3 = self.params['mtx'] @ (cv2.Rodrigues(rvecs[i])[0] @ objpoints[i].T + tvecs[i])
            imgpoints3[0, :] = imgpoints3[0, :] / imgpoints3[2, :]
            imgpoints3[1, :] = imgpoints3[1, :] / imgpoints3[2, :]
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        print("total project error: {}".format(mean_error / len(objpoints)))

        mean_error = 0
        print("verify image points to object points")
        for i in range(len(imgpoints)):
            objpoints2 = self.backProjectPoints(imgpoints[i], rvecs[i], tvecs[i])
            error = cv2.norm(objpoints[i], objpoints2, cv2.NORM_L2) / len(objpoints2)
            mean_error += error
        print("total back project error: {}".format(mean_error / len(imgpoints)))
        return self.params['mtx'], self.params['dist']

    def getAverageMMPD(self, points, real_dist):
        points = points[0].reshape(self.params['chess_size'][0], self.params['chess_size'][1], 2)
        diff_h = points[:, 1:] - points[:, :-1]
        diff_v = points[1:, :] - points[:-1, :]
        dist_h = np.linalg.norm(diff_h, axis=2)
        dist_v = np.linalg.norm(diff_v, axis=2)
        mu_dist_h = np.mean(dist_h)
        mu_dist_v = np.mean(dist_v)
        self.params['mmpd'] = real_dist * 2 / (mu_dist_h + mu_dist_v)
        return self.params['mmpd']

    def getXYByMarkPoint(self, img, cur_xy):
        new_img, center = self.undistort_rotate(img)
        pt, size = detect_mark_point(new_img, 24.7)
        if pt is None or size is None:
            return None, None, None, None
        offset2center = np.array(center) - np.array(pt)
        if self.params['mmpd'] is None:
            self.params['mmpd'] = np.loadtxt(os.path.join('data', 'mmpd.txt'))
        offset2center = offset2center * self.params['mmpd']
        offset2pin = (np.array(self.params['pin_xy']) - np.array(self.params['cam_xy']))
        offset2center[0] = -offset2center[0]
        machine_xy = np.array(cur_xy) + offset2center + offset2pin
        print('machine_xy=', machine_xy)
        return pt, size, machine_xy, new_img

    def findBoardCorners(self, gray):
        patternsize = (self.params['chess_size'][1], self.params['chess_size'][0])
        ret, corners = cv2.findChessboardCorners(gray, patternsize, None)
        if not ret:
            print("findChessboardCorners error!")
            return None, None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        imgp = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return imgp

    def getExteriorParameter(self, img):
        """
        get exterior parameter of a certain img.
        :param img: undistort img
        :return: rvec
        """
        patternsize = (self.params['chess_size'][1], self.params['chess_size'][0])
        objp = np.zeros((patternsize[0] * patternsize[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:patternsize[0], 0:patternsize[1]].T.reshape(-1, 2)
        objp = objp * self.params['grid_length']
        if len(img.shape) == 3 and img.shape[-1] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        imgp = self.findBoardCorners(gray)

        retval, rvec, tvec = cv2.solvePnP(objp, imgp, self.params['mtx'], self.params['dist'])
        if not retval:
            print("solvePnP error!")
            return None, None

        if self.verbose:
            print('rvec=', rvec)
            print('tvec=', tvec)

        self.params['rvec'] = rvec
        self.params['tvec'] = tvec
        return imgp, objp, rvec, tvec
