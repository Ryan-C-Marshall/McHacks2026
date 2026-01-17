from flask import Flask, render_template, send_file, make_response
from flask_socketio import SocketIO, emit
import cv2
import base64
import threading
import time
import io
from util import load_video_thumbnail, load_videos_from_directory

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
    pass

@app.route('/')
def index():
    return render_template('index.html')

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

@socketio.on('stop_tracking')
def handle_stop_tracking():
    global stop_tracking
    stop_tracking = True

if __name__ == '__main__':
    socketio.run(app, debug=True)