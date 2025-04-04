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
                body2d = body_frame.get_body2d(0)
                kpts2d = body2d.numpy()
                kpts2d = kpts2d[self.kinect_to_coco]

                # Annotate the image with the detected landmarks
                cap, annotated_image = capture.get_color_image()
                for i in range(len(kpts2d)):
                    if kpts2d[i][0] < 0 or kpts2d[i][1] < 0:
                        continue
                    cv2.circle(annotated_image, (int(kpts2d[i][0]), int(kpts2d[i][1])), 5, (0, 255, 0), -1)
        
        kpts3d = kpts3d.tolist()
        return annotated_image, kpts3d
    
if __name__ == "__main__":
    coco_skeleton = [
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    detector = KinectDetector()
    key_wait = 10
    with tqdm.tqdm() as pbar:
        while True:
            anno_img, landmarks = detector.get_landmarks(require_annotation=False)
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

    


        
        

