# This file uses MediaPipe, licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import time
from typing import Optional, List, Tuple

import numpy as np

import mediapipe
from mediapipe.tasks import python as python_tasks
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import sys
sys.path.append('D:\jxy\mediapipe-apiserver\mediapipe_apiserver\detector')
from mediapipe_detector import MediaPipeDetector


class MediaPipeMVSDetector:
    def __init__(self) -> None:
        self.left_detector = MediaPipeDetector()
        self.right_detector = MediaPipeDetector()
        self.mp2coco = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    def get_landmarks(self, image_left_np: np.ndarray, image_right_np: np.ndarray, K: np.ndarray, 
                      baseline, require_annotation=False) -> Tuple[Optional[np.ndarray], List[List[Tuple[float, float]]]]:
        annotated_imagel, uvs_left = self.left_detector.get_landmarks(image_left_np, require_annotation=require_annotation)
        annotated_imager, uvs_right = self.right_detector.get_landmarks(image_right_np, require_annotation=require_annotation)

        W, H = image_left_np.shape[1], image_left_np.shape[0]

        uvs_left = np.array(uvs_left[0])[self.mp2coco]  # [33, 4] -> [17, 4]
        uvs_right = np.array(uvs_right[0])[self.mp2coco]    # [33, 4] -> [17, 4]

        conf_left = uvs_left[:, -1:]    # [17, 1]
        conf_right = uvs_right[:, -1:]  # [17, 1]
        conf = (conf_left + conf_right) / 2  # [17, 1]

        lambda_c = conf_left / (conf_left + conf_right)  # [17, 1]
        j3d_local = lambda_c * uvs_left[:, :-1] + (1 - lambda_c) * uvs_right[:, :-1]  # [17, 3]

        uvs_left[:, 0] = uvs_left[:, 0] * W
        uvs_left[:, 1] = uvs_left[:, 1] * H

        uvs_right[:, 0] = uvs_right[:, 0] * W
        uvs_right[:, 1] = uvs_right[:, 1] * H

        # print("uvsl: ", uvsl)
        depth = K[0, 0] * baseline / (uvs_left[:, 0] - uvs_right[:, 0])
        x_c_left = (uvs_left[:, 0] - K[0, 2]) * depth / K[0, 0]
        y_c_left = (uvs_left[:, 1] - K[1, 2]) * depth / K[1, 1]

        x_c_right = (uvs_right[:, 0] - K[0, 2]) * depth / K[0, 0]
        y_c_right = (uvs_right[:, 1] - K[1, 2]) * depth / K[1, 1]
        
        x_c = (x_c_left + x_c_right) / 2
        y_c = (y_c_left + y_c_right) / 2

        j3d_cam = np.concatenate((x_c[:, np.newaxis], y_c[:, np.newaxis], depth[:, np.newaxis], conf), axis=1)
        j3d_local = np.concatenate((j3d_local, conf), axis=1)  # [17, 4]

        return j3d_cam, j3d_local

if __name__ == '__main__':
    import pyzed.sl as sl
    import cv2
    zed = sl.Camera()
    init_params = sl.InitParameters()

    # Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    W = 1920
    H = 1080

    # https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1UNIT.html
    init_params.coordinate_units = sl.UNIT.METER  # Set coordinate units

    init_params.depth_mode = sl.DEPTH_MODE.NONE

    # https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1COORDINATE__SYSTEM.html
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # image_l = sl.Mat()
    # image_r = sl.Mat()
    image_s = sl.Mat()

    calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
    baseline = calibration_params.get_camera_baseline()
    left_cam = calibration_params.left_cam
    right_cam = calibration_params.right_cam
    print("left_cam: ", "fx: ", left_cam.fx, "fy: ", left_cam.fy, "cx: ", left_cam.cx, "cy: ", left_cam.cy)
    print("right_cam: ", "fx: ", right_cam.fx, "fy: ", right_cam.fy, "cx: ", right_cam.cx, "cy: ", right_cam.cy)
    print("baseline: ", baseline)
    # exit()

    left_detector = MediaPipeDetector()
    right_detector = MediaPipeDetector()

    import matplotlib.pyplot as plt
    import tqdm
    import time

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    time.sleep(5)
    start_time = time.time()
    print("start")
    j3ds = []
    j3d_lefts = []
    j3d_rights = []
    with tqdm.tqdm() as pbar:
        while True:
            pbar.update(1)
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                # Retrieve left image
                zed.retrieve_image(image_s, sl.VIEW.SIDE_BY_SIDE)

                image_s_np = image_s.get_data()
                image_l_np = image_s_np[:, :W, :]
                image_r_np = image_s_np[:, W:, :]
                # image_r_np = image_r.get_data(deep_copy=True)

                image_l_np = cv2.cvtColor(image_l_np, cv2.COLOR_BGRA2RGB)
                image_r_np = cv2.cvtColor(image_r_np, cv2.COLOR_BGRA2RGB)

                annotated_imagel, uvsl = left_detector.get_landmarks(image_l_np, require_annotation=True)
                annotated_imager, uvsr = right_detector.get_landmarks(image_r_np, require_annotation=True)


                if len(uvsl) == 0:
                    uvsl = np.random.rand(33, 4)
                else:
                    uvsl = np.array(uvsl[0])
                if len(uvsr) == 0:
                    uvsr = np.random.rand(33, 4)
                else:
                    uvsr = np.array(uvsr[0])

                j3d_local = uvsl.copy()
                uvsl[:, 0] = uvsl[:, 0] * W
                uvsl[:, 1] = uvsl[:, 1] * H

                uvsr[:, 0] = uvsr[:, 0] * W
                uvsr[:, 1] = uvsr[:, 1] * H

                K_left = np.asarray([[left_cam.fx,        0.0,       left_cam.cx],
                                    [0.0,        left_cam.fy,       left_cam.cy],
                                    [0.0,                0.0,               1.0]])
                K_right = np.asarray([[right_cam.fx,        0.0,       right_cam.cx],
                                    [0.0,        right_cam.fy,       right_cam.cy],
                                    [0.0,                0.0,               1.0]])

                # print("uvsl: ", uvsl)
                depth_left = K_left[0, 0] * baseline / (uvsl[:, 0] - uvsr[:, 0])
                x_c_left = (uvsl[:, 0] - K_left[0, 2]) * depth_left / K_left[0, 0]
                y_c_left = (uvsl[:, 1] - K_left[1, 2]) * depth_left / K_left[1, 1]

                depth_right = K_right[0, 0] * baseline / (uvsl[:, 0] - uvsr[:, 0])
                x_c_right = (uvsr[:, 0] - K_right[0, 2]) * depth_right / K_right[0, 0]
                y_c_right = (uvsr[:, 1] - K_right[1, 2]) * depth_right / K_right[1, 1]
                
                x_c = (x_c_left + x_c_right) / 2
                y_c = (y_c_left + y_c_right) / 2
                depth = (depth_left + depth_right) / 2
                j3d = np.concatenate((x_c[:, np.newaxis], y_c[:, np.newaxis], depth[:, np.newaxis]), axis=1)
                j3d_left = np.concatenate((x_c_left[:, np.newaxis], y_c_left[:, np.newaxis], depth_left[:, np.newaxis]), axis=1)
                j3d_right = np.concatenate((x_c_right[:, np.newaxis], y_c_right[:, np.newaxis], depth_right[:, np.newaxis]), axis=1)
                # j3ds.append(j3d)
                # j3d_lefts.append(j3d_left)
                # j3d_rights.append(j3d_right)

                if True:
                    # draw the pose landmarks on image using cv2
                    for i in range(len(uvsl)):
                        cv2.circle(image_l_np, (int(uvsl[i][0]), int(uvsl[i][1])), 5, (0, 255, 0), -1)
                    for i in range(len(uvsr)):
                        cv2.circle(image_r_np, (int(uvsr[i][0]), int(uvsr[i][1])), 5, (0, 255, 0), -1)

                    # concat 2 images
                    # image = np.concatenate((image_l_np, image_r_np), axis=1)
                    # concat 2 annotated images
                    anno_image = np.concatenate((annotated_imagel, annotated_imager), axis=1)

                    # image = np.concatenate((image, anno_image), axis=0)
                    cv2.imshow("image", anno_image)
                    cv2.waitKey(1)

                # y轴最大最小差值
                # print("y_c: ", y_c.max() - y_c.min())
                # print(depth.mean())

                if True:
                    ax.cla()
                    # 对每一个点 alpha 根据 visibility 进行设置，大小大一点
                    vis = uvsl[:, 3]
                    for i in range(len(uvsl)):
                        # ax.scatter(x_c[i], y_c[i], depth[i], c='r', s=100, alpha=vis[i])
                        ax.scatter(j3d_local[:, 0], j3d_local[:, 1], j3d_local[:, 2], c='b', s=100, alpha=0.5)
                
                    # label
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_xlim(-1, 1)
                    ax.set_ylim(-1, 1)
                    ax.set_zlim(1, 3)
                    plt.title('3D Pose')
                    plt.draw()
                    plt.pause(0.001)

            # if time.time() - start_time > 20:
            #     print("end")
            #     break
    
    # np.save("j3d.npy", np.array(j3ds))
    # np.save("j3d_left.npy", np.array(j3d_lefts))
    # np.save("j3d_right.npy", np.array(j3d_rights))
