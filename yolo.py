import cv2
import random
from ultralytics import YOLO


yolo = YOLO("yolov8n.pt")

def getColours(cls_num):
    """Generate unique colors for each class ID"""
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

video = cv2.VideoCapture("echo1.mp4")
if not video.isOpened():
    print("Could not open video")
    exit()

frame_count = 0
# Create a resizable window and set its size
cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Tracking', 1280, 720)  # Width, Height in pixels

while True:
    ret, frame = video.read()
    if not ret:
        break
    results = yolo.track(frame, stream=True)

    for result in results:
        class_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cls = int(box.cls[0])
                class_name = class_names[cls]

                conf = float(box.conf[0])

                colour = getColours(cls)

                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                cv2.putText(frame, f"{class_name} {conf:.2f}",
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, colour, 2)
    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break

video.release()
