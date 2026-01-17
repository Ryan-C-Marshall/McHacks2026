import os
from flask import Flask, render_template, send_file, make_response
from flask_socketio import SocketIO, emit
import cv2
import base64
import threading
from flask import Flask, render_template
from flask_socketio import SocketIO
from cv_tracker import pick_tracker

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

VIDEO_PATH = "echo1.mp4"
FPS_LIMIT = 30

BOX_SIZE = 100
tracker_num = 1
tracker_type = tracker_types[tracker_num]

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

tracking_active = False
stream_thread = None

# --- tracker state ---
clicked_pt = None
tracker = None
tracker_inited = False


def make_square_bbox(center, box_size, w, h):
    cx, cy = center
    half = box_size // 2
    x = int(cx - half)
    y = int(cy - half)
    x = max(0, min(x, w - box_size))
    y = max(0, min(y, h - box_size))
    return (x, y, int(box_size), int(box_size))


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("select_point")
def on_select_point(data):
    global clicked_pt, tracker_inited, tracker
    clicked_pt = (int(data["x"]), int(data["y"]))
    tracker_inited = False
    tracker = None
    socketio.emit("status", f"Point received: {clicked_pt}. Tracker will initialize on next frame.")


def stream_video():
    global tracking_active, tracker, tracker_inited, clicked_pt

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        socketio.emit("status", "Error: Cannot open video")
        return

    socketio.emit("status", "Streaming started (click the video to select ROI center)")

    while tracking_active:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Optional: choose a fixed output size (or keep native)
        TARGET_W, TARGET_H = 640, 480
        frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)

        h, w = frame.shape[:2]

        # Initialize tracker once we have a click
        if (clicked_pt is not None) and (not tracker_inited):
            # If your click comes in at different scale than the frame, you must scale it here.
            bbox = make_square_bbox(clicked_pt, BOX_SIZE, w, h)
            tracker = pick_tracker(tracker_type)
            tracker.init(frame, bbox)
            tracker_inited = True
            socketio.emit("status", f"Tracker initialized using {tracker_type}")

        # Update tracker if initialized
        if tracker_inited and tracker is not None:
            timer = cv2.getTickCount()

            ok, bbox = tracker.update(frame)

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
            if ok:
                x, y, bw, bh = bbox
                cx = int(x + bw / 2)
                cy = int(y + bh / 2)
                cv2.drawMarker(
                    frame, (cx, cy), (0, 255, 0),
                    markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2
                )
            else:
                cv2.putText(frame, "Tracking failure", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

             # Display tracker type on frame
            cv2.putText(frame, tracker_type + " Tracker", (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,170,50),2);
        
            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,170,50), 2);

        # Encode and emit frame
        ok_jpg, buffer = cv2.imencode(".jpg", frame)
        if ok_jpg:
            frame_b64 = base64.b64encode(buffer).decode("utf-8")
            socketio.emit("frame", frame_b64)

        socketio.sleep(1 / FPS_LIMIT)

    cap.release()
    socketio.emit("status", "Streaming stopped")
def track_and_emit(video_path):
    pass


@app.route('/landing')
def landing():
    videos = load_videos_from_directory()
    return render_template('landing_page.html', videos=videos)

@app.route('/thumbnail/<path:video_path>')
def thumbnail(video_path):
    frame = load_video_thumbnail(video_path)
    if frame is None:
        return "Error loading thumbnail", 404
    _, buffer = cv2.imencode('.jpg', frame)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/jpeg'
    return response

@socketio.on('start_tracking')
def handle_start_tracking():
    global tracking_thread, stop_tracking
    if tracking_thread and tracking_thread.is_alive():
        return
    stop_tracking = False
    video_path = "videos/Echo/echo1.mp4"  # Hardcoded for now
    tracking_thread = threading.Thread(target=track_and_emit, args=(video_path,))
    tracking_thread.start()
    print("Started tracking thread.")
@socketio.on("start_tracking")
def start_tracking():
    global tracking_active, stream_thread
    if not tracking_active:
        tracking_active = True
        stream_thread = threading.Thread(target=stream_video, daemon=True)
        stream_thread.start()


@socketio.on("stop_tracking")
def stop_tracking():
    global tracking_active
    tracking_active = False


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)