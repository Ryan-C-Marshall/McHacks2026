import threading
from flask import Flask, render_template, make_response, request
from flask_socketio import SocketIO
import cv2

from util import load_video_thumbnail, load_videos_from_directory, stream_video, DEFAULT_VIDEO_PATH

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

FPS_LIMIT = 30
TARGET_SIZE = (640, 480)

BOX_SIZE = 100
tracker_num = 1
tracker_type = tracker_types[tracker_num]

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# One shared state object for util.stream_video
state = {
    "tracking_active": False,
    "clicked_pt": None,
    "tracker": None,
    "tracker_inited": False,
    "show_bbox": True,
    "resume_frame": 0,
    "video_path": DEFAULT_VIDEO_PATH,
}

stream_thread = None


@app.route("/")
def index():
    state["video_path"] = request.args.get("video", DEFAULT_VIDEO_PATH)

    return render_template("index.html")


@app.route("/landing")
def landing():
    videos = load_videos_from_directory()
    return render_template("landing_page.html", videos=videos)


@app.route("/thumbnail/<path:video_path>")
def thumbnail(video_path):
    frame = load_video_thumbnail(video_path)
    if frame is None:
        return "Error loading thumbnail", 404
    _, buffer = cv2.imencode(".jpg", frame)
    resp = make_response(buffer.tobytes())
    resp.headers["Content-Type"] = "image/jpeg"
    return resp


@socketio.on("select_point")
def on_select_point(data):
    state["clicked_pt"] = (int(data["x"]), int(data["y"]))
    state["tracker_inited"] = False
    state["tracker"] = None
    socketio.emit("status", f"Point received: {state['clicked_pt']}. Tracker will initialize on next frame.")


@socketio.on("toggle_bbox")
def on_toggle_bbox(data):
    state["show_bbox"] = bool(data.get("show", True))
    socketio.emit("status", f"Show bbox: {state['show_bbox']}")


@socketio.on("start_tracking")
def start_tracking():
    global stream_thread

    if state["tracking_active"]:
        return

    state["tracking_active"] = True

    # If you want to use the landing page selection later, pass that path instead.

    stream_thread = threading.Thread(
        target=stream_video,
        kwargs={
            "socketio": socketio,
            "fps_limit": FPS_LIMIT,
            "target_size": TARGET_SIZE,
            "box_size": BOX_SIZE,
            "tracker_type": tracker_type,
            "state": state,
        },
        daemon=True,
    )
    stream_thread.start()
    socketio.emit("status", f"Started streaming: {state['video_path']}")


@socketio.on("pause_tracking")
def pause_tracking():
    state["tracking_active"] = False
    socketio.emit("status", "Pausing...")


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)