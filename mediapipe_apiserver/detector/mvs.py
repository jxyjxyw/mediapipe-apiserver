import time
from typing import Optional, List, Tuple

import numpy as np
import pyzed.sl as sl
import cv2

import sys
sys.path.append('D:\jxy\mediapipe-apiserver\mediapipe_apiserver\detector')
from mmposedetector import MMPoseDetector


class MVSDetector:
    zed38_to_coco = [5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    K = np.asarray([[524.575439453125,  0.0,       634.0914916992188],
                    [0.0,        524.575439453125,  357.988037109375],
                    [0.0,               0.0,                     1.0]])

    def __init__(self) -> None:
        self.zed = sl.Camera()
        init_params = sl.InitParameters()

        # Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA
        init_params.camera_resolution = sl.RESOLUTION.HD720

        # https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1UNIT.html
        init_params.coordinate_units = sl.UNIT.METER  # Set coordinate units

        init_params.depth_mode = sl.DEPTH_MODE.NONE

        # https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1COORDINATE__SYSTEM.html
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

        # Open the camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        self.image_l = sl.Mat()
        self.image_r = sl.Mat()
        
        calibration_params = self.zed.get_camera_information().camera_configuration.calibration_parameters
        self.baseline = calibration_params.get_camera_baseline()

        self.mmpose_detector = MMPoseDetector()

    def get_landmarks(self, require_annotation=False):
        landmarks = np.random.rand(17, 4)  # [17, 4] for 17 keypoints, x, y, z, conf
        annotated_image = None
        
        # Grab an image
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            cam_timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
            # Retrieve image
            self.zed.retrieve_image(self.image_l, sl.VIEW.LEFT)
            self.zed.retrieve_image(self.image_r, sl.VIEW.RIGHT)  # Retrieve right image
            # self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)     # Retrieve depth matrix. Depth is aligned on the left RGB image

            imagel_np = self.image_l.get_data()[:, :, :3]  # [720, 1280, 4] -> [720, 1280, 3]
            imager_np = self.image_r.get_data()[:, :, :3]  # [720, 1280, 4] -> [720, 1280, 3]

            annotated_imagel, uvsl, scoresl = self.mmpose_detector.get_landmarks(imagel_np, require_annotation=require_annotation)
            annotated_imager, uvsr, scoresr = self.mmpose_detector.get_landmarks(imager_np, require_annotation=require_annotation)

            depth = self.K[0, 0] * self.baseline / (uvsl[:, 0] - uvsr[:, 0])

            # clip to [0, 1279] and [0, 719]
            uvsl = np.clip(uvsl, 0, [1279, 719])
            # check num of nans in depth_np
            # print("Num of nans in depth_np: ", np.isnan(depth_np).sum())

            # depth = depth_np[uvs[:, 1].astype(int), uvs[:, 0].astype(int)]

            x_c = (uvsl[:, 0] - self.K[0, 2]) * depth / self.K[0, 0]
            y_c = (uvsl[:, 1] - self.K[1, 2]) * depth / self.K[1, 1]

            print("y_c: ", y_c.max() - y_c.min())

            landmarks = np.concatenate((x_c[:, np.newaxis], y_c[:, np.newaxis], depth[:, np.newaxis], scoresl[:, np.newaxis]), axis=1)
        
        landmarks = landmarks.tolist()
        return annotated_imagel, landmarks, cam_timestamp
    
if __name__ == "__main__":
    coco_skeleton = [
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    import matplotlib.pyplot as plt
    import tqdm
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    detector = MVSDetector()
    key_wait = 10
    with tqdm.tqdm() as pbar:
        while True:
            pbar.update()
            anno_img, landmarks, _ = detector.get_landmarks(require_annotation=False)
            # print(len(landmarks))
            # time.sleep(0.5)
            if anno_img is not None:
                cv2.imshow("Annotated Image", anno_img)
                key = cv2.waitKey(key_wait)
                if key == 113: # for 'q' key
                    print("Exiting...")
                    break
                if key == 109: # for 'm' key
                    if (key_wait>0):
                        print("Pause")
                        key_wait = 0 
                    else : 
                        print("Restart")
                        key_wait = 10
        
        # print(np.asarray(landmarks))

        # plt 2D
        # if True:
        #     uvs = np.array(landmarks)[:, :2]
        #     ax.cla()
        #     ax.scatter(uvs[:, 0], uvs[:, 1], c='b', marker='o')
        #     # plt skeleton
        #     for i, j in coco_skeleton:
        #         ax.plot([uvs[i][0], uvs[j][0]], [uvs[i][1], uvs[j][1]], c='r')
        #     plt.xlim(0, 1280)
        #     plt.ylim(720, 0)
        #     # label
        #     ax.set_xlabel('X')
        #     ax.set_ylabel('Y')
        #     plt.title('2D Pose')
        #     plt.draw()
        #     plt.pause(0.001)

            # plt 3D
            # if True:
            #     landmarks = np.array(landmarks)
            #     ax.cla()
            #     ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], c='b', marker='o')
            #     # plt skeleton
            #     for i, j in coco_skeleton:
            #         ax.plot([landmarks[i][0], landmarks[j][0]], [landmarks[i][1], landmarks[j][1]], [landmarks[i][2], landmarks[j][2]], c='r')
            
            #     # label
            #     ax.set_xlabel('X')
            #     ax.set_ylabel('Y')
            #     ax.set_zlabel('Z')
            #     ax.set_xlim(-1, 1)
            #     ax.set_ylim(-1, 1)
            #     ax.set_zlim(1, 3)
            #     plt.title('3D Pose')
            #     plt.draw()
            #     plt.pause(0.001)

        
        if False:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            landmarks = np.array(landmarks)
            print(landmarks)
            ax.cla()
            ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], c='b', marker='o')
            # plt skeleton
            for i, j in coco_skeleton:
                ax.plot([landmarks[i][0], landmarks[j][0]], [landmarks[i][1], landmarks[j][1]], [landmarks[i][2], landmarks[j][2]], c='r')

            # label
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            plt.draw()
            plt.pause(0.1)


        
        

