try:
    from .mediapipe_detector import MediaPipeDetector
except:
    print("MediaPipeDetector is not available")

# try:
from .mmposedetector import MMPoseDetector
# except Exception as e:
#     print(e)
#     print("MMPoseDetector is not available")

from .zed2 import Zed2Detector
# from .kinect import KinectDetector
from .zed_mmpose import Zed2MMPoseDetector
