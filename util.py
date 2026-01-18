import cv2
import base64
import threading

from tracker_fusion import draw_tracked_boxes

STATE_LOCK = threading.RLock()


DEFAULT_VIDEO_PATH = "videos/Echo/echo1.mp4"

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

def pick_tracker(tracker_type):

    if int(minor_ver) < 3:
        return cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            return cv2.legacy.TrackerBoosting_create()
        if tracker_type == 'MIL':
            return cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            return cv2.legacy.TrackerKCF_create()
        if tracker_type == 'TLD':
            return cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            return cv2.legacy.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            return cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            return cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            return cv2.TrackerCSRT_create()

def load_video_thumbnail(video_path, time_sec=0):
    """Load a thumbnail image from the video at the specified time (in seconds)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_number = int(fps * time_sec)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Error: Could not read frame.")
        return None

    return frame


def load_videos_from_directory(directory_path='videos/'):
    """Load all video file paths from the specified directory."""
    import os
    video_list = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(root, file)
                title = os.path.splitext(file)[0].replace('_', ' ').title()
                desc = "Description for " + title
                video_list.append({'title': title, 'desc': desc, 'path': video_path})

    return video_list


def make_square_bbox(center, box_size, w, h):
    cx, cy = center
    half = box_size // 2
    x = int(cx - half)
    y = int(cy - half)
    x = max(0, min(x, w - box_size))
    y = max(0, min(y, h - box_size))
    return (x, y, int(box_size), int(box_size))

def generate_tracker_colour(tracker_num):
    colours = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
    ]
    return colours[tracker_num % len(colours)]

def add_text_to_tracker(tracker_obj, text, position):
    tracker_obj["texts"].append((text, position))

def add_arrow_to_tracker(tracker_obj, start_point, end_point):
    tracker_obj["arrows"].append((start_point, end_point))

def create_new_tracker(state, tracker_type, box_size, frame, socketio):
    h, w = frame.shape[:2]

    with STATE_LOCK:
        pt = state.get("clicked_pt")
        state["clicked_pt"] = None

    if pt is None:
        return

    bbox0 = make_square_bbox(pt, box_size, w, h)

    ensemble_types = state.get("ensemble_types")

    tracker_group = {
        "subtrackers": [],      # list[ {name, tracker, ok, bbox} ]
        "last_bbox": bbox0,     # consensus bbox will live here
        "texts": [],
        "arrows": [],
        "colour": generate_tracker_colour(len(state["trackers"])),
    }

    # Create and init each algorithm with the same bbox
    for name in ensemble_types:
        tr = pick_tracker(name)
        tr.init(frame, bbox0)
        ok, bb = tr.update(frame)  # “prime” it like your prototype
        tracker_group["subtrackers"].append({
            "name": name,
            "tracker": tr,
            "ok": bool(ok),
            "bbox": bb if ok else bbox0,
        })

    # If at least one subtracker is OK, seed last_bbox from the “best available”
    ok_list = [t for t in tracker_group["subtrackers"] if t["ok"]]
    if ok_list:
        tracker_group["last_bbox"] = ok_list[0]["bbox"]

    with STATE_LOCK:
        state["trackers"].append(tracker_group)

    socketio.emit("status", f"Tracker group initialized with: {ensemble_types}")
    
def delete_tracker(state, tracker_num):
    with STATE_LOCK:
        if 0 <= tracker_num < len(state["trackers"]):
            state["trackers"].pop(tracker_num)

def delete_all_trackers(state):
    with STATE_LOCK:
        state["trackers"] = []

def update_tracker(state, frame, tracker_type, tracker_num, paused=False):
    with STATE_LOCK:
        if tracker_num >= len(state.get("trackers", [])):
            return
        tracker_obj = state["trackers"][tracker_num]

    fps = None

    subtrackers = tracker_obj["subtrackers"]

    # Update each subtracker (unless paused)
    if not paused:
        timer = cv2.getTickCount()
        for t in subtrackers:
            tr = t.get("tracker")
            if tr is None:
                t["ok"] = False
                continue
            ok, bb = tr.update(frame)
            t["ok"] = bool(ok)
            t["bbox"] = bb
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    mean_p1, mean_p2 = draw_tracked_boxes(frame, tracker_obj["subtrackers"])

    # Save consensus bbox as this ROI’s anchor for text/arrows
    with STATE_LOCK:
        tracker_obj["last_bbox"] = (float(mean_p1[0]), float(mean_p1[1]),
                                    float(mean_p2[0] - mean_p1[0]), float(mean_p2[1] - mean_p1[1]))

    cons_w = float(mean_p2[0] - mean_p1[0])
    cons_h = float(mean_p2[1] - mean_p1[1])

    cx = int(mean_p1[0] + cons_w / 2.0)
    cy = int(mean_p1[1] + cons_h / 2.0)

    cv2.drawMarker(
        frame,
        (cx, cy),
        tracker_obj["colour"],
        markerType=cv2.MARKER_CROSS,
        markerSize=14,
        thickness=2,
    )
    # Draw consensus box
    if state.get("show_bbox", True):
        cv2.rectangle(frame, mean_p1, mean_p2, tracker_obj["colour"], 2)

    # Now draw your existing overlay items using the consensus bbox as the origin
    x, y, bw, bh = tracker_obj["last_bbox"]

    overlay_colour = tracker_obj.get("colour", (0, 255, 255))

    # Texts (relative offsets)
    for text, (dx, dy) in tracker_obj.get("texts", []):
        cv2.putText(frame, text, (int(x) + dx, int(y) + dy + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, overlay_colour, 2)

    # Arrows (relative offsets)
    for (sx, sy), (ex, ey) in tracker_obj.get("arrows", []):
        cv2.arrowedLine(frame, (int(x) + sx, int(y) + sy),
                        (int(x) + ex, int(y) + ey),
                        overlay_colour, 2, tipLength=0.3)

    if tracker_num == 0 and fps is not None:
        cv2.putText(
            frame, f"FPS : {int(fps)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 170, 50), 2
        )


def stream_video(
    socketio,
    fps_limit: int,
    target_size: tuple[int, int],
    box_size: int,
    tracker_type: str,
    state: dict,
):
    video_path = state.get("video_path", DEFAULT_VIDEO_PATH)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        socketio.emit("status", f"Error: Cannot open video: {video_path}")
        with STATE_LOCK:
            state["paused"] = False
        return

    target_w, target_h = target_size

    # Seek to last known position (resume)
    resume_frame = int(state.get("resume_frame", 0) or 0)
    if resume_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, resume_frame)
        socketio.emit("status", f"Streaming resumed")
    else:
        socketio.emit("status", "Streaming started")

    # Initialize frame
    ret, frame = cap.read()

    while True:
        with STATE_LOCK:
            state["streaming_video"] = True

        if(not state.get("tracking_active")):
            break

        if not state.get("paused", True):
            # If not paused, update video
            ret, frame = cap.read()

            if not ret or frame is None:
                state["paused"] = True
                break

            # Record resume position after successful read
            state["resume_frame"] = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Resize to tracking/output size
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Initialize tracker on a click
        if state.get("clicked_pt") is not None:
            create_new_tracker(state, tracker_type, box_size, frame, socketio)
            with STATE_LOCK:
                state["clicked_pt"] = None  # consume click

        # Update trackers (if initialized)
        with STATE_LOCK:
            tracker_indices = list(range(len(state.get("trackers", []))))

        for i in tracker_indices:
            update_tracker(state, frame, tracker_type, tracker_num=i, paused=state.get("paused", False))

        # Encode and emit frame
        ok_jpg, buffer = cv2.imencode(".jpg", frame)
        if ok_jpg:
            frame_b64 = base64.b64encode(buffer).decode("utf-8")
            socketio.emit("frame", frame_b64)

        socketio.sleep(1 / fps_limit)
    
    cap.release()
    socketio.emit("status", "Streaming stopped")
    with STATE_LOCK:
        state["streaming_video"] = False
        