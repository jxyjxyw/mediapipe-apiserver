try:
    from .mediapipe import MediaPipeDetector
except:
    print("MediaPipeDetector is not available")

try:
    from .mmpose import MMPoseDetector
except:
    print("MMPoseDetector is not available")
