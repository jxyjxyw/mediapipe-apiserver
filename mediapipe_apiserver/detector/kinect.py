import cv2
import numpy as np
import pykinect_azure as pykinect
import tqdm

class KinectDetector:
    kinect_to_coco = [27, 28, 30, 29, 31, 5, 12, 6, 13, 7, 14, 18, 22, 19, 23, 20, 24]

    def __init__(self) -> None:
        # Initialize the library, if the library is not found, add the library path as argument
        pykinect.initialize_libraries(track_body=True)

        # Modify camera configuration
        device_config = pykinect.default_configuration
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        #print(device_config)

        # Start device
        self.device = pykinect.start_device(config=device_config)

        # Start body tracker
        self.bodyTracker = pykinect.start_body_tracker()


    def get_landmarks(self, require_annotation=False):
        kpts3d = np.random.rand(17, 4)  # [17, 4] for 17 keypoints, x, y, z, conf
        annotated_image = None
        
        # Get capture
        capture = self.device.update()

        # Get body tracker frame
        body_frame = self.bodyTracker.update()

        if body_frame.get_num_bodies() > 0:
            body = body_frame.get_body(0)	# Get the first body
            joints = body.numpy()   # [n_kpts, 8]
            kpts3d = joints[self.kinect_to_coco] # [17, 4]
            kpts3d = kpts3d[:, [0, 1, 2, 7]] / 1000.0   # mm to m

            # fill nan with random values
            kpts3d[np.isnan(kpts3d)] = np.random.rand(np.sum(np.isnan(kpts3d)))

            if require_annotation:

                # Annotate the image with the detected landmarks
                cap, annotated_image = capture.get_color_image()
                color_skeleton = body_frame.draw_bodies(annotated_image, pykinect.K4A_CALIBRATION_TYPE_COLOR)
        
        kpts3d = kpts3d.tolist()
        return annotated_image, kpts3d
    
if __name__ == "__main__":
    def create_skeleton(array):
    # 创建一个空的点云
        point_cloud = o3d.geometry.PointCloud()

        # 创建骨架的线段
        lines = []
        colors = []

        # 创建球体表示每个关节
        spheres = []
        
        for i, point in enumerate(array):
            # 创建球体表示关节
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sphere.translate(point)
            spheres.append(sphere)

        # 为骨骼中的每条连接添加线段
        for (start, end) in coco_skeleton:
            lines.append([start, end])
            colors.append([1, 0, 0])  # 设置线条颜色为红色

        # 创建线段
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(array)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return spheres, line_set

    def update_visualization(array):
        # 清空之前的几何体
        vis.clear_geometries()
        
        # 创建骨架模型
        spheres, line_set = create_skeleton(array)

        # 添加球体和线段到可视化中
        for sphere in spheres:
            vis.add_geometry(sphere)
        
        vis.add_geometry(line_set)

        # 创建坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        vis.add_geometry(coordinate_frame)

        # 更新可视化窗口
        vis.poll_events()
        vis.update_renderer()

    import open3d as o3d
    coco_skeleton = [
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    detector = KinectDetector()
    key_wait = 10

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    with tqdm.tqdm() as pbar:
        while True:
            anno_img, landmarks = detector.get_landmarks(require_annotation=False)

            p3d = np.array(landmarks)[:, :3]
            update_visualization(p3d)

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
            pbar.update(1)

    


        
        

