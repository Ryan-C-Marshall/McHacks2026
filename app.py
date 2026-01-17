from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import base64
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for tracking
tracking_thread = None
stop_tracking = False

FRAME_RESIZE = 8
BOX_SIZE = 100

def make_square_bbox(center, box_size, w, h):
    cx, cy = center
    half = box_size // 2
    x = int(cx - half)
    y = int(cy - half)
    x = max(0, min(x, w - box_size))
    y = max(0, min(y, h - box_size))
    return (x, y, int(box_size), int(box_size))

def track_and_emit(video_path):
    global stop_tracking
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[7]

    if int(cv2.__version__.split('.')[1]) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'CSRT':
            tracker = cv2.TrackerCSRT_create()

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        socketio.emit('status', 'Could not open video')
        return

    ok, frame = video.read()
    if not ok:
        socketio.emit('status', 'Cannot read video file')
        return

    frame = cv2.resize(frame, None, fx=FRAME_RESIZE, fy=FRAME_RESIZE, interpolation=cv2.INTER_LINEAR)
    h, w = frame.shape[:2]

    # Use a fixed center for simplicity (you can modify to accept from frontend)
    center = (w // 2, h // 2)  # Center of frame
    bbox = make_square_bbox(center, BOX_SIZE, w, h)

    ok = tracker.init(frame, bbox)
    if not ok:
        socketio.emit('status', 'Tracker initialization failed')
        return

    socketio.emit('status', 'Tracking started')

    while not stop_tracking:
        ok, frame = video.read()
        if not ok:
            break

        frame = cv2.resize(frame, None, fx=FRAME_RESIZE, fy=FRAME_RESIZE, interpolation=cv2.INTER_LINEAR)

        timer = cv2.getTickCount()
        ok, bbox = tracker.update(frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        if ok:
            x, y, bw, bh = bbox
            cx = int(x + bw / 2)
            cy = int(y + bh / 2)
            cv2.drawMarker(frame, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
        else:
            cv2.putText(frame, "Tracking failure detected", (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(frame, tracker_type + " Tracker", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 170, 50), 2)
        cv2.putText(frame, "FPS : " + str(int(fps)), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 170, 50), 2)

        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('frame', frame_base64)

        time.sleep(0.05)  # Adjust for frame rate

    video.release()
    socketio.emit('status', 'Tracking stopped')

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_tracking')
def handle_start_tracking():
    global tracking_thread, stop_tracking
    if tracking_thread and tracking_thread.is_alive():
        return
    stop_tracking = False
    video_path = "videos/Echo/echo1.mp4"  # Hardcoded for now
    tracking_thread = threading.Thread(target=track_and_emit, args=(video_path,))
    tracking_thread.start()

@socketio.on('stop_tracking')
def handle_stop_tracking():
    global stop_tracking
    stop_tracking = True

if __name__ == '__main__':
    socketio.run(app, debug=True)