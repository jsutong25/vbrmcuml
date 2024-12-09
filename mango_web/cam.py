import cv2

for i in range(10):  # Check up to 10 devices
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available.")
        cap.release()