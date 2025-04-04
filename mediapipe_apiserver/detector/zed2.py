import time
from typing import Optional, List, Tuple

import numpy as np
import pyzed.sl as sl
import cv2

class Zed2Detector:
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

        init_params.depth_mode = sl.DEPTH_MODE.ULTRA

        # https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1COORDINATE__SYSTEM.html
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

        # Open the camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        # Enable Positional tracking (mandatory for object detection)
        positional_tracking_parameters = sl.PositionalTrackingParameters()
        # If the camera is static, uncomment the following line to have better performances
        # positional_tracking_parameters.set_as_static = True
        self.zed.enable_positional_tracking(positional_tracking_parameters)

        body_param = sl.BodyTrackingParameters()
        body_param.enable_tracking = True                # Track people across images flow
        body_param.enable_body_fitting = True            # Smooth skeleton move
        body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST     # 3 choices
        body_param.body_format = sl.BODY_FORMAT.BODY_38  # Choose the BODY_FORMAT you wish to use (17, 34, 38)

        # Enable Object Detection module
        self.zed.enable_body_tracking(body_param)

        self.body_runtime_param = sl.BodyTrackingRuntimeParameters()
        self.body_runtime_param.detection_confidence_threshold = 40

        # Get ZED camera information
        camera_info = self.zed.get_camera_information()
        # 2D viewer utilities
        self.display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280), min(camera_info.camera_configuration.resolution.height, 720))

        self.bodies = sl.Bodies()
        self.image = sl.Mat()

    def get_landmarks(self, require_annotation=False):
        landmarks = np.random.rand(17, 4)  # [17, 4] for 17 keypoints, x, y, z, conf
        annotated_image = None
        
        # Grab an image
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT, sl.MEM.CPU, self.display_resolution)
            # Retrieve bodies
            self.zed.retrieve_bodies(self.bodies, self.body_runtime_param)

            if len(self.bodies.body_list) > 0:
                obj = self.bodies.body_list[0]
                kpts3d = obj.keypoint   # [n_kpts, 3]
                # fill nan with random values
                kpts3d[np.isnan(kpts3d)] = np.random.rand(np.sum(np.isnan(kpts3d)))
                
                kpts2d = obj.keypoint_2d[self.zed38_to_coco]    # [n_kpts, 2]

                # calculate kpts2d from kpts3d
                # kpts2d = np.matmul(self.K, kpts3d.T).T
                # kpts2d = kpts2d[:, :2] / kpts2d[:, 2:3]
                # kpts2d = kpts2d[:, :2]  # [n_kpts, 2]
                
                conf = obj.keypoint_confidence.reshape(-1, 1)  # [n_kpts, 1]
                conf /= 100.0   # confidence of Zed 2 is in [0, 100]
                conf[np.isnan(conf)] = 0.0  # fill nan with 0.0

                # concat x, y, z, conf to landmarks
                landmarks = np.concatenate([kpts3d, conf], axis=1)
                landmarks = landmarks[self.zed38_to_coco]  # [17, 4]

                if require_annotation:
                    # Annotate the image with the detected landmarks
                    annotated_image = np.copy(self.image.get_data())
                    for i in range(len(kpts2d)):
                        if kpts2d[i][0] < 0 or kpts2d[i][1] < 0:
                            continue
                        cv2.circle(annotated_image, (int(kpts2d[i][0]), int(kpts2d[i][1])), 5, (0, 255, 0), -1)
        
        landmarks = landmarks.tolist()
        return annotated_image, landmarks
    
if __name__ == "__main__":
    coco_skeleton = [
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    

    detector = Zed2Detector()
    key_wait = 10
    while True:
        anno_img, landmarks = detector.get_landmarks(require_annotation=True)
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
        
        # print(landmarks)
        if True:
            landmarks = np.array(landmarks)
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


        
        

