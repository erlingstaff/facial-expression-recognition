# Facial Expression Recognition Example

OpenCV & DeepFace facial recognition and emotion detection

Takes in an .mp4 video file and streams it with along with facial recognition and tries its best to tell which emotion the person in the video is showing.

Only every 5th frame is rendered, as deepface recognition is very expensive and this an alternative version of this example is intended to run on a server with limited resources.

stop = q