import time
from typing import Optional, List, Tuple

import numpy as np
import pyzed.sl as sl
import cv2

import sys
sys.path.append('D:\jxy\mediapipe-apiserver\mediapipe_apiserver\detector')
from mmposedetector import MMPoseDetector

def get_depth(depth: np.ndarray, uv_list: np.ndarray) -> np.ndarray:
    """
    对每个 uv，如果本身是有限值，就直接返回；
    否则不断向外按半径扩展，直到找到最近的有限深度值（非NaN且非Inf）。
    如果全图都没有有限值，则返回 NaN。
    
    Args:
        depth: (H, W) numpy数组
        uv_list: (N, 2) numpy数组
    Returns:
        (N,) numpy数组，每个元素是对应 uv 的“最近有限深度”或 NaN。
    """
    h, w = depth.shape
    results = np.empty(len(uv_list), dtype=depth.dtype)

    # 最大半径：扩展到图像最远角为止
    max_radius = max(h, w)

    for idx, (u, v) in enumerate(uv_list):
        u_int, v_int = int(u), int(v)
        # 越界直接 NaN
        if not (0 <= u_int < w and 0 <= v_int < h):
            results[idx] = np.nan
            continue

        center_val = depth[v_int, u_int]
        if np.isfinite(center_val):
            # 如果中心点是有限值，直接用它
            results[idx] = center_val
            continue

        found = False
        # 从半径1开始，四条边扫描
        for r in range(1, max_radius + 1):
            # 扫描上下边
            for du in range(-r, r + 1):
                for dv in (-r, r):
                    uu, vv = u_int + du, v_int + dv
                    if 0 <= uu < w and 0 <= vv < h:
                        val = depth[vv, uu]
                        if np.isfinite(val):
                            results[idx] = val
                            found = True
                            break
                if found:
                    break
            if found:
                break

            # 扫描左右边（排除已扫描的角点）
            for dv in range(-r + 1, r):
                for du in (-r, r):
                    uu, vv = u_int + du, v_int + dv
                    if 0 <= uu < w and 0 <= vv < h:
                        val = depth[vv, uu]
                        if np.isfinite(val):
                            results[idx] = val
                            found = True
                            break
                if found:
                    break
            if found:
                break

        # 如果整图都没有限值，返回 NaN
        if not found:
            results[idx] = np.nan

    return results


class Zed2MMPoseDetector:
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

        init_params.depth_mode = sl.DEPTH_MODE.NEURAL

        # https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1COORDINATE__SYSTEM.html
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

        # Open the camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        self.imagel = sl.Mat()
        self.depth = sl.Mat()

        self.mmpose_detector = MMPoseDetector()

    def get_landmarks(self, require_annotation=False):
        landmarks = np.random.rand(17, 4)  # [17, 4] for 17 keypoints, x, y, z, conf
        annotated_image = None
        cam_timestamp = None
        
        # Grab an image
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            cam_timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
            # Retrieve left image
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)     # Retrieve depth matrix. Depth is aligned on the left RGB image

            image_np = self.image.get_data()[:, :, :3]  # [720, 1280, 4] -> [720, 1280, 3]
            depth_np = self.depth.get_data()    # [720, 1280]

            annotated_image, uvs, scores = self.mmpose_detector.get_landmarks(image_np, require_annotation=require_annotation)

            if uvs is not None:
                # clip to [0, 1279] and [0, 719]
                uvs = np.clip(uvs, 0, [1279, 719])
                # check num of nans in depth_np
                # print("Num of nans in depth_np: ", np.isnan(depth_np).sum())

                # depth = depth_np[uvs[:, 1].astype(int), uvs[:, 0].astype(int)]
                depth = get_depth(depth_np, uvs)

                x_c = (uvs[:, 0] - self.K[0, 2]) * depth / self.K[0, 0]
                y_c = (uvs[:, 1] - self.K[1, 2]) * depth / self.K[1, 1]

                landmarks = np.concatenate((x_c[:, np.newaxis], y_c[:, np.newaxis], depth[:, np.newaxis], scores[:, np.newaxis]), axis=1)
        
        landmarks = landmarks.tolist()
        return annotated_image, landmarks, cam_timestamp
    
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

    detector = Zed2MMPoseDetector()
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
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(1, 3)
                plt.title('3D Pose')
                plt.draw()
                plt.pause(0.001)

        
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


        
        

