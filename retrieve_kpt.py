import numpy as np
import sys
from mediapipe_apiserver.detector.mmposedetector import MMPoseDetector
from mediapipe_apiserver.detector.mediapipe_detector import MediaPipeDetector
import torch
import os
import tqdm

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

def get_landmarks_from_rgbd(image_np, depth_np, K, mmpose_detector, require_annotation=False):
    _, uvs, scores = mmpose_detector.get_landmarks(image_np, require_annotation=require_annotation)

    uvs = np.clip(uvs, 0, [1279, 719])

    depth = get_depth(depth_np, uvs)

    x_c = (uvs[:, 0] - K[0, 2]) * depth / K[0, 0]
    y_c = (uvs[:, 1] - K[1, 2]) * depth / K[1, 1]

    landmarks = np.concatenate((x_c[:, np.newaxis], y_c[:, np.newaxis], depth[:, np.newaxis], scores[:, np.newaxis]), axis=1)

    return landmarks

def retrieve(record_path):
    record = torch.load(record_path)
    K = np.asarray([[524.575439453125,  0.0,       634.0914916992188],
                    [0.0,        524.575439453125,  357.988037109375],
                    [0.0,               0.0,                     1.0]])
    kpts = []
    for i in tqdm.trange(len(record['image'])):
        image = record['image'][i]
        depth = record['depth'][i]

        # to numpy
        image_np = image.cpu().numpy()
        depth_np = depth.cpu().numpy()

        landmarks = get_landmarks_from_rgbd(image_np, depth_np, K, mmpose_detector, require_annotation=False)
        kpts.append(landmarks)
    
    # save kpts to file
    kpts = np.array(kpts)  # (N, 17, 4)

    # to tensor
    kpts = torch.from_numpy(kpts).float()
    # save
    save_path = os.path.join(os.path.dirname(record_path), 'cocokpts.pt')
    torch.save(kpts, save_path)
    print(f"save kpts to {save_path}")

if __name__ == '__main__':
    mmpose_detector = MMPoseDetector()
    record_path = r"D:\jxy\Robustcap-ttt\fake_data\20250429_1641_test10s\record.pt"
    
    retrieve(record_path)
