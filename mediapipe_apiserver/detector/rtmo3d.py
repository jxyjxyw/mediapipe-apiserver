# tune multi-threading params
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(0)

# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser
from typing import List

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import inference_topdown, init_model, _inference_topdown_batch
from mmpose.registry import VISUALIZERS
from mmpose.structures import (PoseDataSample, merge_data_samples,
                               split_instances)
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.visualization import Pose3dLocalVisualizer

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from rtmpose3d import *  # noqa: F401, F403

import tqdm
import torch
import pyzed.sl as sl

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument(
        'pose3d_estimator_config',
        type=str,
        default=None,
        help='Config file for the 3D pose estimator')
    parser.add_argument(
        'pose3d_estimator_checkpoint',
        type=str,
        default=None,
        help='Checkpoint file for the 3D pose estimator')
    parser.add_argument('--input', type=str, default='', help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='Whether to show visualizations')
    parser.add_argument(
        '--disable-rebase-keypoint',
        action='store_true',
        default=False,
        help='Whether to disable rebasing the predicted 3D pose so its '
        'lowest keypoint has a height of 0 (landing on the ground). Rebase '
        'is useful for visualization when the model do not predict the '
        'global position of the 3D pose.')
    parser.add_argument(
        '--disable-norm-pose-2d',
        action='store_true',
        default=False,
        help='Whether to scale the bbox (along with the 2D pose) to the '
        'average bbox scale of the dataset, and move the bbox (along with the '
        '2D pose) to the average bbox center of the dataset. This is useful '
        'when bbox is small, especially in multi-person scenarios.')
    parser.add_argument(
        '--num-instances',
        type=int,
        default=1,
        help='The number of 3D poses to be visualized in every frame. If '
        'less than 0, it will be set to the number of pose results in the '
        'first frame.')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='Whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.5,
        help='Bounding box score threshold')
    parser.add_argument('--kpt-thr', type=float, default=0.3)
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='Inference mode. If set to True, can not use future frame'
        'information when using multi frames for inference in the 2D pose'
        'detection stage. Default: False.')

    args = parser.parse_args()
    return args

class RTM3DDetector:
    def __init__(self):
        self.baseline = 0.1201544851064682
        self.K = np.asarray([[524.575439453125,  0.0,       634.0914916992188],
                             [0.0,        524.575439453125,  357.988037109375],
                             [0.0,               0.0,                     1.0]])
        
        self.detector = init_detector(
            r"D:\jxy\mmpose\projects\rtmpose3d\configs\rtmdet_m_640-8xb32_coco-person.py", 
            r"D:\jxy\mmpose\projects\rtmpose3d\demo\rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth", 
            device='cuda:0')
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        self.pose_estimator = init_model(
            r"D:\jxy\mmpose\projects\rtmpose3d\configs\rtmw3d-x_8xb32_cocktail14-384x288.py",
            r"D:\jxy\mmpose\projects\rtmpose3d\demo\rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth",
            device='cuda:0')
        self.pose_estimator.cfg.model.test_cfg.mode = 'vis'

        self.frame_idx = 0
        self.next_id = 0
        self.pose_est_results_list = []
        self.pose_est_results_last = []

    def get_landmarks(self, frame: np.ndarray) -> np.ndarray:
        (pose_est_results, self.pose_est_results_list, pred_3d_instances,
        self.next_id) = process_one_image(
            args=None,
            detector=self.detector,
            frame=frame,
            frame_idx=self.frame_idx,
            pose_estimator=self.pose_estimator,
            pose_est_results_last=self.pose_est_results_last,
            pose_est_results_list=self.pose_est_results_list,
            next_id=self.next_id,
            visualize_frame=None,
            visualizer=None
            )
        
        self.pose_est_results_last = pose_est_results
        self.frame_idx += 1

        landmarks = torch.randn(17, 4)

        if len(pred_3d_instances) <= 0:
            return landmarks
        
        # print(len(pred_3d_instances.keypoints))
        kpt3d_simcc = pred_3d_instances.keypoints[0, :17]     # [17, 3]
        scores = pred_3d_instances.keypoint_scores[0, :17]    # [17, ]

        landmarks = np.concatenate((kpt3d_simcc, scores[:, np.newaxis]), axis=1)  # [17, 4]
        return landmarks
    
    def get_landmarks_online(self, frame: np.ndarray) -> np.ndarray:
        (pose_est_results, self.pose_est_results_list, pred_3d_instances,
        self.next_id) = process_batch_image(
            args=None,
            detector=self.detector,
            frame=frame,
            frame_idx=self.frame_idx,
            pose_estimator=self.pose_estimator,
            pose_est_results_last=self.pose_est_results_last,
            pose_est_results_list=self.pose_est_results_list,
            next_id=self.next_id,
            visualize_frame=None,
            visualizer=None
            )
        
        self.pose_est_results_last = pose_est_results
        self.frame_idx += 1

        landmarks = torch.randn(17, 4)

        if len(pred_3d_instances) <= 0:
            return landmarks
        
        assert len(pred_3d_instances) == 2

        kpt3d_simcc_left = pred_3d_instances.keypoints[0, :17]     # [17, 3]
        scores_left = pred_3d_instances.keypoint_scores[0, :17]    # [17, ]
        kpt2d_left = pred_3d_instances.transformed_keypoints[0, :17]  # [17, 2]

        kpt3d_simcc_right = pred_3d_instances.keypoints[1, :17]     # [17, 3]
        scores_right = pred_3d_instances.keypoint_scores[1, :17]    # [17, ]
        kpt2d_right = pred_3d_instances.transformed_keypoints[1, :17]  # [17, 2]

        conf = (scores_left + scores_right) / 2.0  # [17, ]
        lambda_c = scores_left / (scores_left + scores_right)
        lambda_c = lambda_c[:, np.newaxis]  # [17, 1]
        kpt3d_simcc = lambda_c * kpt3d_simcc_left + (1 - lambda_c) * kpt3d_simcc_right  # [17, 3]

        if False:
            kpt3d_simcc_root = (kpt3d_simcc[11:12, :] + kpt3d_simcc[12:13, :]) / 2    # [1, 3]
            nose_root_distance = np.linalg.norm(kpt3d_simcc[0:1, :] - kpt3d_simcc_root, axis=-1, keepdims=True)
            scale = 0.7 / nose_root_distance   # 0.57 meter
            kpt3d_simcc -= kpt3d_simcc_root
            kpt3d_simcc = kpt3d_simcc * scale

        kpt3d_simcc = np.concatenate((kpt3d_simcc, conf[:, np.newaxis]), axis=1)  # [17, 4]

        # ===============================

        depth = self.K[0, 0] * self.baseline / (kpt2d_left[:, 0] - kpt2d_right[:, 0])

        kpt2d_left = np.clip(kpt2d_left, 0, [1279, 719])

        x_c = (kpt2d_left[:, 0] - self.K[0, 2]) * depth / self.K[0, 0]
        y_c = (kpt2d_left[:, 1] - self.K[1, 2]) * depth / self.K[1, 1]

        kpt3d_cam = np.concatenate((x_c[:, np.newaxis], y_c[:, np.newaxis], depth[:, np.newaxis], conf[:, np.newaxis]), axis=1)  # [17, 4]

        return kpt3d_cam, kpt3d_simcc

def process_one_image(args, detector, frame: np.ndarray, frame_idx: int,
                      pose_estimator,
                      pose_est_results_last: List[PoseDataSample],
                      pose_est_results_list: List[List[PoseDataSample]],
                      next_id: int, visualize_frame: np.ndarray,
                      visualizer: Pose3dLocalVisualizer):
    # pose_dataset = pose_estimator.cfg.test_dataloader.dataset
    pose_det_dataset_name = pose_estimator.dataset_meta['dataset_name']

    # First stage: conduct 2D pose detection in a Topdown manner
    # use detector to obtain person bounding boxes
    det_result = inference_detector(detector, frame)
    pred_instance = det_result.pred_instances.cpu().numpy()

    # filter out the person instances with category and bbox threshold
    # e.g. 0 for person in COCO
    bboxes = pred_instance.bboxes
    # bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
    #                                pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                   pred_instance.scores > 0.5)]

    # estimate pose results for current image
    pose_est_results = inference_topdown(pose_estimator, frame, bboxes)

    # post-processing
    for idx, pose_est_result in enumerate(pose_est_results):
        pose_est_result.track_id = pose_est_results[idx].get('track_id', 1e4)

        pred_instances = pose_est_result.pred_instances
        keypoints = pred_instances.keypoints
        keypoint_scores = pred_instances.keypoint_scores
        if keypoint_scores.ndim == 3:
            keypoint_scores = np.squeeze(keypoint_scores, axis=1)
            pose_est_results[
                idx].pred_instances.keypoint_scores = keypoint_scores
        if keypoints.ndim == 4:
            keypoints = np.squeeze(keypoints, axis=1)

        pose_est_results[idx].pred_instances.keypoints = keypoints

    pose_est_results = sorted(
        pose_est_results, key=lambda x: x.get('track_id', 1e4))

    pred_3d_data_samples = merge_data_samples(pose_est_results)
    pred_3d_instances = pred_3d_data_samples.get('pred_instances', None)

    return pose_est_results, pose_est_results_list, pred_3d_instances, next_id

def process_batch_image(args, detector, frame: np.ndarray, frame_idx: int,
                      pose_estimator,
                      pose_est_results_last: List[PoseDataSample],
                      pose_est_results_list: List[List[PoseDataSample]],
                      next_id: int, visualize_frame: np.ndarray,
                      visualizer: Pose3dLocalVisualizer):
    # pose_dataset = pose_estimator.cfg.test_dataloader.dataset
    pose_det_dataset_name = pose_estimator.dataset_meta['dataset_name']

    # First stage: conduct 2D pose detection in a Topdown manner
    # use detector to obtain person bounding boxes
    det_result = inference_detector(detector, frame)
    bbox_list = []

    for batch_idx in range(len(frame)):
        pred_instance = det_result[batch_idx].pred_instances.cpu().numpy()

        # filter out the person instances with category and bbox threshold
        # e.g. 0 for person in COCO
        bboxes = pred_instance.bboxes
        # bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
        #                                pred_instance.scores > args.bbox_thr)]
        bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                    pred_instance.scores > 0.5)]
        bbox_list.append(bboxes)

    # estimate pose results for current image
    pose_est_results = _inference_topdown_batch(pose_estimator, frame, bbox_list)

    # post-processing
    for idx, pose_est_result in enumerate(pose_est_results):
        pose_est_result.track_id = pose_est_results[idx].get('track_id', 1e4)

        pred_instances = pose_est_result.pred_instances
        keypoints = pred_instances.keypoints
        keypoint_scores = pred_instances.keypoint_scores
        if keypoint_scores.ndim == 3:
            keypoint_scores = np.squeeze(keypoint_scores, axis=1)
            pose_est_results[
                idx].pred_instances.keypoint_scores = keypoint_scores
        if keypoints.ndim == 4:
            keypoints = np.squeeze(keypoints, axis=1)

        pose_est_results[idx].pred_instances.keypoints = keypoints

    pose_est_results = sorted(
        pose_est_results, key=lambda x: x.get('track_id', 1e4))

    pred_3d_data_samples = merge_data_samples(pose_est_results)
    pred_3d_instances = pred_3d_data_samples.get('pred_instances', None)

    return pose_est_results, pose_est_results_list, pred_3d_instances, next_id


def main():
    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parse_args()

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    pose_estimator = init_model(
        args.pose3d_estimator_config,
        args.pose3d_estimator_checkpoint,
        device=args.device.lower())

    det_kpt_color = pose_estimator.dataset_meta.get('keypoint_colors', None)
    det_dataset_skeleton = pose_estimator.dataset_meta.get(
        'skeleton_links', None)
    det_dataset_link_color = pose_estimator.dataset_meta.get(
        'skeleton_link_colors', None)

    pose_estimator.cfg.model.test_cfg.mode = 'vis'
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.line_width = args.thickness
    pose_estimator.cfg.visualizer.det_kpt_color = det_kpt_color
    pose_estimator.cfg.visualizer.det_dataset_skeleton = det_dataset_skeleton
    pose_estimator.cfg.visualizer.det_dataset_link_color = det_dataset_link_color  # noqa: E501
    pose_estimator.cfg.visualizer.skeleton = det_dataset_skeleton
    pose_estimator.cfg.visualizer.link_color = det_dataset_link_color
    pose_estimator.cfg.visualizer.kpt_color = det_kpt_color
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)

    if args.input == 'webcam':
        input_type = 'webcam'
    else:
        input_type = mimetypes.guess_type(args.input)[0].split('/')[0]

    if args.output_root == '':
        save_output = False
    else:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'
        save_output = True

    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 3D plt
    if PLOT_3D:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        coco_skeleton = [
            (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
            (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]

    pose_est_results_list = []
    pred_instances_list = []
    if input_type == 'image':
        frame = mmcv.imread(args.input, channel_order='rgb')
        _, _, pred_3d_instances, _ = process_one_image(
            args=args,
            detector=detector,
            frame=args.input,
            frame_idx=0,
            pose_estimator=pose_estimator,
            pose_est_results_last=[],
            pose_est_results_list=pose_est_results_list,
            next_id=0,
            visualize_frame=frame,
            visualizer=visualizer)
        
        kpt2d = pred_3d_instances.transformed_keypoints[0, :17]     # [17, 3]
        coco3d = pred_3d_instances.keypoints[0, :17]                 # [17, 3]

        # print(kpt2d)

        # print(coco3d)

        if PLOT_3D:
            # plt 3D in matplotlib
            ax.cla()
            # set axis lim
            x_lim = np.max(coco3d[:, 0]) - np.min(coco3d[:, 0])
            y_lim = np.max(coco3d[:, 1]) - np.min(coco3d[:, 1])
            z_lim = np.max(coco3d[:, 2]) - np.min(coco3d[:, 2])
            x_mean = np.mean(coco3d[:, 0])
            y_mean = np.mean(coco3d[:, 1])
            z_mean = np.mean(coco3d[:, 2])
            ax.set_xlim(x_mean - x_lim, x_mean + x_lim)
            ax.set_ylim(y_mean - y_lim, y_mean + y_lim)
            ax.set_zlim(z_mean - z_lim, z_mean + z_lim)
            ax.scatter(coco3d[:, 0], coco3d[:, 1], coco3d[:, 2], c='b', marker='o')
            
            for i in range(len(coco_skeleton)):
                ax.plot3D(*zip(coco3d[coco_skeleton[i][0]], coco3d[coco_skeleton[i][1]]), color='b', linewidth=2)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            plt.pause(0.01)
            plt.show()

        if args.save_predictions:
            # save prediction results
            pred_instances_list = split_instances(pred_3d_instances)

        if save_output:
            frame_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(frame_vis), output_file)

    elif input_type in ['webcam', 'video']:
        next_id = 0
        pose_est_results = []

        if args.input == 'webcam':
            video = cv2.VideoCapture(0)
        else:
            video = cv2.VideoCapture(args.input)

        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        if int(major_ver) < 3:
            fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps = video.get(cv2.CAP_PROP_FPS)
        print(f'Video FPS: {fps}')

        video_writer = None
        frame_idx = 0

        
        
        # =============== ZED camera ================
        zed = sl.Camera()
        init_params = sl.InitParameters()

        # Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA
        init_params.camera_resolution = sl.RESOLUTION.HD720

        # https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1UNIT.html
        init_params.coordinate_units = sl.UNIT.METER  # Set coordinate units

        init_params.depth_mode = sl.DEPTH_MODE.NONE

        # https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1COORDINATE__SYSTEM.html
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

        # Open the camera
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        image = sl.Mat()
        # ============================================

        with tqdm.tqdm() as pbar:
            while video.isOpened():
                # success, frame = video.read()
                # frame = frame[:, :frame.shape[1] // 2, :]
                frame_idx += 1

                # if not success:
                #     break
                if zed.grab() == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_image(image, sl.VIEW.LEFT)

                frame = image.get_data()[:, :, :3]

                pose_est_results_last = pose_est_results

                # First stage: 2D pose detection
                # make person results for current image
                (pose_est_results, pose_est_results_list, pred_3d_instances,
                next_id) = process_one_image(
                    args=args,
                    detector=detector,
                    frame=frame,
                    frame_idx=frame_idx,
                    pose_estimator=pose_estimator,
                    pose_est_results_last=pose_est_results_last,
                    pose_est_results_list=pose_est_results_list,
                    next_id=next_id,
                    # visualize_frame=mmcv.bgr2rgb(frame),
                    visualize_frame=None,
                    # visualizer=visualizer,
                    visualizer=None
                    )
                
                # print(f'[DEBUG] pose_est_results: {pose_est_results[0].pred_instances.transformed_keypoints[0, :17]}')
                kpt2d = pred_3d_instances.transformed_keypoints[0, :17]     # [17, 3]
                coco3d = pred_3d_instances.keypoints[0, :17]                 # [17, 3]

                print(kpt2d)

                print(coco3d)
                
                if PLOT_3D:
                    # plt 3D in matplotlib
                    ax.cla()
                    ax.set_xlim(-400, 0)
                    ax.set_ylim(-400, 0)
                    ax.set_zlim(-0, 400)
                    ax.scatter(coco3d[:, 0], coco3d[:, 1], coco3d[:, 2], c='b', marker='o')
                    
                    for i in range(len(coco_skeleton)):
                        ax.plot3D(*zip(coco3d[coco_skeleton[i][0]], coco3d[coco_skeleton[i][1]]), color='b', linewidth=2)

                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')

                    ax.set_title(f'Frame {frame_idx}')
                    plt.pause(0.01)
                    plt.draw()


                # plt kpt2d on frame
                # for i in range(kpt2d.shape[0]):
                #     cv2.circle(frame, (int(kpt2d[i, 0]), int(kpt2d[i, 1])), 3, (0, 255, 0), -1)
                # cv2.imshow('frame', frame)

                # cv2.waitKey(1)

                # if args.save_predictions:
                #     # save prediction results
                #     pred_instances_list.append(
                #         dict(
                #             frame_id=frame_idx,
                #             instances=split_instances(pred_3d_instances)))

                # if save_output:
                #     frame_vis = visualizer.get_image()
                #     if video_writer is None:
                #         # the size of the image with visualization may vary
                #         # depending on the presence of heatmaps
                #         video_writer = cv2.VideoWriter(output_file, fourcc, fps,
                #                                     (frame_vis.shape[1],
                #                                         frame_vis.shape[0]))
                #     video_writer.write(mmcv.rgb2bgr(frame_vis))

                # if args.show:
                #     # press ESC to exit
                #     if cv2.waitKey(5) & 0xFF == 27:
                #         break
                #     time.sleep(args.show_interval)

                pbar.update(1)

        video.release()

        if video_writer:
            video_writer.release()
    else:
        args.save_predictions = False
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')

    # if args.save_predictions:
    #     with open(args.pred_save_path, 'w') as f:
    #         json.dump(
    #             dict(
    #                 meta_info=pose_estimator.dataset_meta,
    #                 instance_info=pred_instances_list),
    #             f,
    #             indent='\t')
    #     print(f'predictions have been saved at {args.pred_save_path}')

    # if save_output:
    #     input_type = input_type.replace('webcam', 'video')
    #     print_log(
    #         f'the output {input_type} has been saved at {output_file}',
    #         logger='current',
    #         level=logging.INFO)


if __name__ == '__main__':
    coco_skeleton = [
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    zed = sl.Camera()
    init_params = sl.InitParameters()

    # Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA
    init_params.camera_resolution = sl.RESOLUTION.HD720

    # https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1UNIT.html
    init_params.coordinate_units = sl.UNIT.METER  # Set coordinate units

    init_params.depth_mode = sl.DEPTH_MODE.NONE

    # https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1COORDINATE__SYSTEM.html
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    image = sl.Mat()

    detector = RTM3DDetector()

    calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
    baseline = calibration_params.get_camera_baseline()
    print(f'baseline: {baseline}')

    frame = [np.random.rand(720, 1280, 3) * 255 for _ in range(2)]
    with tqdm.tqdm() as pbar:
        while True:
            # if zed.grab() == sl.ERROR_CODE.SUCCESS:
            #     zed.retrieve_image(image, sl.VIEW.SIDE_BY_SIDE)
            # frame = image.get_data()[:, :, :3]
            # frame = [frame[:, :1280].copy(), frame[:, 1280:].copy()]
            
            kpts_cam, kpts_simcc = detector.get_landmarks_online(frame)

            # print(kpts_simcc[5, 2] - kpts_simcc[9, 2])
            # print(max(kpts_simcc[:, 1]) - min(kpts_simcc[:, 1]))

            if True:
                ax.cla()
                ax.scatter(kpts_simcc[:, 0], kpts_simcc[:, 1], kpts_simcc[:, 2], c='b', marker='o')
                # plt skeleton
                for i, j in coco_skeleton:
                    ax.plot([kpts_simcc[i][0], kpts_simcc[j][0]], [kpts_simcc[i][1], kpts_simcc[j][1]], [kpts_simcc[i][2], kpts_simcc[j][2]], c='r')
            
                # label
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)
                plt.title('3D Pose')
                plt.draw()
                plt.pause(0.001)


            pbar.update(1)
