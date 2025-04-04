try:
    from .mediapipe import MediaPipeDetector
except:
    print("MediaPipeDetector is not available")

# try:
from .mmpose import MMPoseDetector
# except Exception as e:
#     print(e)
#     print("MMPoseDetector is not available")

from .zed2 import Zed2Detector
from .kinect import KinectDetector
