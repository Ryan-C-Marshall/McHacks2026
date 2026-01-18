import threading
from flask import Flask, render_template, make_response, request
from flask_socketio import SocketIO
import cv2

from util import BoxTracker, load_video_thumbnail, load_videos_from_directory, stream_video, DEFAULT_VIDEO_PATH, delete_tracker, STATE_LOCK, add_arrow_to_tracker, add_text_to_tracker, Tracker, PolygonTracker

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

FPS_LIMIT = 30
TARGET_SIZE = (640, 480)

BOX_SIZE = 100

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# type 'tracker': a dict with 'tracker', 'tracker_inited', 'text', 'text_position', 'arrow', 'arrow_position'

# One shared state object for util.stream_video
state = {
    "tracking_active": False,
    "streaming_video": False,
    "paused": False,
    "clicked_pt": None,
    "create_new_tracker_type": None, # "BOX", "LINE", "POLYGON"
    "add_point_to_polygon_tracker": None, # (tracker_idx, (x,y))
    "trackers": [],     # tracker, tracker_inited
    "show_bbox": False,
    "resume_frame": 0,
    "video_path": None,
    "ensemble_types": ["KCF", "CSRT", "MEDIANFLOW"],
}

stream_thread = None

def _bbox_offsets_from_abs_click(tracker_obj: Tracker, x: int, y: int):
    """Convert absolute frame coords (x,y) to offsets relative to latest bbox.
    Returns {"closest_tracker_index": index, "offsets": (dx, dy)} or None if bbox not available.
    """
    if isinstance(tracker_obj, PolygonTracker):
        min_dist = float('inf')
        closest_idx = -1
        for i, point in enumerate(tracker_obj.points):
            bx, by, bw, bh = point.last_bbox
            cx = bx + bw / 2
            cy = by + bh / 2
            dist = (x - cx)**2 + (y - cy)**2
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        if closest_idx != -1:
            bx, by, _, _ = tracker_obj.points[closest_idx].last_bbox
            dx = x - bx
            dy = y - by
            return {"closest_tracker_index": closest_idx, "offsets": (dx, dy)}
    elif isinstance(tracker_obj, BoxTracker):
        # For BoxTracker
        bx, by, _, _ = tracker_obj.last_bbox
        dx = x - bx
        dy = y - by
        return {"closest_tracker_index": 0, "offsets": (dx, dy)}


@app.route("/")
def landing():
    videos = load_videos_from_directory()
    return render_template("landing_page.html", videos=videos)


@app.route("/index")
def index():
    print("Loading index page...")

    # end any existing tracking thread
    with STATE_LOCK:
        state["tracking_active"] = False
    while state.get("streaming_video", True):
        # wait for streaming to stop
        continue

    socketio.emit("status", "Ending tracking thread...")

    # start the new tracking session
    with STATE_LOCK:
        state["trackers"] = []
        state["clicked_pt"] = None
        state["create_new_tracker_type"] = None
        state["tracking_active"] = False
        state["paused"] = False
        state["resume_frame"] = 0
        state["show_bbox"] = True
        state["video_path"] = request.args.get("video", DEFAULT_VIDEO_PATH)
    start_tracking()
    return render_template("index.html")


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
    with STATE_LOCK:
        state["clicked_pt"] = (int(data["x"]), int(data["y"]))
        state["create_new_tracker_type"] = "BOX"
    socketio.emit("status", f"Point received: {state['clicked_pt']}. Tracker {state['create_new_tracker_type']} will initialize on next frame.")

@socketio.on("start_polygon")
def on_start_polygon():
    with STATE_LOCK:
        state["create_new_tracker_type"] = "POLYGON"

@socketio.on("select_polygon")
def on_select_polygon(data):
    with STATE_LOCK:
        idx = int(data.get("tracker_num"))

        if idx >= len(state.get("trackers", [])):
            socketio.emit("status", f"Polygon not added (tracker {idx} not found)")
            return
        
        state["add_point_to_polygon_tracker"] = (idx, (int(data["x"]), int(data["y"])))
    socketio.emit("status", f"Polygon selected. Tracker {state['create_new_tracker_type']} will initialize on next frame.")

@socketio.on("toggle_bbox")
def on_toggle_bbox(data):
    state["show_bbox"] = bool(data.get("show", True))
    socketio.emit("status", f"Show bbox: {state['show_bbox']}")

@socketio.on("start_tracking")
def start_tracking():
    global stream_thread

    with STATE_LOCK:
        state["paused"] = False
        

    if state["tracking_active"]:
        return

    with STATE_LOCK:
        state["tracking_active"] = True

    stream_thread = threading.Thread(
        target=stream_video,
        kwargs={
            "socketio": socketio,
            "fps_limit": FPS_LIMIT,
            "target_size": TARGET_SIZE,
            "box_size": BOX_SIZE,
            "state": state,
        },
        daemon=True,
    )
    stream_thread.start()
    socketio.emit("status", f"Started streaming: {state['video_path']}")


@socketio.on("pause_tracking")
def pause_tracking():
    state["paused"] = True
    socketio.emit("status", "Pausing...")

@socketio.on("delete_tracker")
def handle_delete(data):
    idx = data.get("tracker_num")

    with STATE_LOCK:
        delete_tracker(state, idx)

@socketio.on("add_text")
def handle_add_text(data):
    idx = int(data.get("tracker_num", -1))
    x = int(data.get("x"))
    y = int(data.get("y"))
    text = str(data.get("text", "")).strip()
    if idx < 0 or not text:
        socketio.emit("status", "Text not added (missing tracker or empty text)")
        return

    with STATE_LOCK:
        if idx >= len(state.get("trackers", [])):
            socketio.emit("status", f"Text not added (tracker {idx} not found)")
            return
        tracker_obj = state["trackers"][idx]
        offs = _bbox_offsets_from_abs_click(tracker_obj, x, y)
        add_text_to_tracker(tracker_obj, text, offs)

    socketio.emit("status", f"Added text to tracker {idx}")


@socketio.on("add_arrow")
def handle_add_arrow(data):
    idx = int(data.get("tracker_num"))
    start = data.get("start") or {}
    end = data.get("end") or {}
    sx, sy = int(start.get("x")), int(start.get("y"))
    ex, ey = int(end.get("x")), int(end.get("y"))
    if idx < 0:
        socketio.emit("status", "Arrow not added (no tracker selected)")
        return

    with STATE_LOCK:
        if idx >= len(state.get("trackers", [])):
            socketio.emit("status", f"Arrow not added (tracker {idx} not found)")
            return
        tracker_obj = state["trackers"][idx]
        s_offs = _bbox_offsets_from_abs_click(tracker_obj, sx, sy)
        e_offs = _bbox_offsets_from_abs_click(tracker_obj, ex, ey)
        add_arrow_to_tracker(tracker_obj, s_offs, e_offs)

    socketio.emit("status", f"Added arrow to tracker {idx}")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)