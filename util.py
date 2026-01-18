import cv2
import base64
import threading

from tracker_fusion import draw_tracked_boxes

STATE_LOCK = threading.RLock()


DEFAULT_VIDEO_PATH = "videos/Echo/echo1.mp4"

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

class FundamentalTracker():
    def __init__(self, tracker_type: str, frame, bbox0):
        self.tracker_type = tracker_type
        self.tracker = pick_tracker(tracker_type)
        self.ok = False
        self.bbox = bbox0
        self.tracker.init(frame, bbox0)

class Tracker():
    def __init__(self, state, box_size, pt, frame):
        self.tracker_num = len(state["trackers"])

        h, w = frame.shape[:2]

        self.bbox0 = make_square_bbox(pt, box_size, w, h)
        self.last_bbox = self.bbox0

        self.colour = generate_tracker_colour(self.tracker_num)

        self.texts = []
        self.arrows = []
        
        self.update(state, frame)
        self.tracker_inited = True
    
    def update(self, state, frame):
        # To be implemented in subclasses
        pass
        
class BoxTracker(Tracker):
    def __init__(self, state, box_size, pt, frame):
        h, w = frame.shape[:2]

        self.bbox0 = make_square_bbox(pt, box_size, w, h)
        
        self.fundamental_trackers: list[FundamentalTracker] = []

        # Initialize fundamental trackers
        self.fundamental_trackers.append(FundamentalTracker("KCF", frame, self.bbox0))
        self.fundamental_trackers.append(FundamentalTracker("CSRT", frame, self.bbox0))
        self.fundamental_trackers.append(FundamentalTracker("MEDIANFLOW", frame, self.bbox0))

        super().__init__(state, box_size, pt, frame)

    def update(self, state, frame):
        update_box_tracker(state, frame, self)

class LineTracker(Tracker):
    def __init__(self):
        pass

class PolygonTracker(Tracker):
    def __init__(self):
        pass

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
    if tracker_type not in ["BOX", "LINE", "POLYGON"]:
        socketio.emit("status", f"Error: Unknown tracker type: {tracker_type}")
        return
    
    with STATE_LOCK:
        pt = state.get("clicked_pt")
        state["clicked_pt"] = None
    
    if pt is None:
        return
        
    new_tracker: Tracker | None = None
    if tracker_type == "BOX":
        new_tracker = BoxTracker(state, box_size, pt, frame)
    elif tracker_type == "LINE":
        new_tracker = LineTracker() # TODO
    elif tracker_type == "POLYGON":
        new_tracker = PolygonTracker() # TODO
    
    if not new_tracker:
        socketio.emit("status", f"Error: Could not create tracker of type: {tracker_type}")
        return

    with STATE_LOCK:
        state["trackers"].append(new_tracker)

    socketio.emit("status", f"Tracker initialized ({new_tracker.tracker_num}): {new_tracker.tracker_inited}")


def delete_tracker(state, tracker_num):
    with STATE_LOCK:
        if 0 <= tracker_num < len(state["trackers"]):
            state["trackers"].pop(tracker_num)

def delete_all_trackers(state):
    with STATE_LOCK:
        state["trackers"] = []

def update_box_tracker(state, frame, tracker_obj: BoxTracker):

    subtrackers = tracker_obj.fundamental_trackers
    paused = state.get("paused", False)

    # Update each subtracker (unless paused)
    if not paused:
        for t in subtrackers:
            tr = t.tracker
            if tr is None:
                t.ok = False
                continue
            ok, bb = tr.update(frame)
            t.ok = bool(ok)
            t.bbox = bb

    outputs = draw_tracked_boxes(frame, subtrackers)
    if outputs is None:
        return
    mean_p1, mean_p2 = outputs

    # Save consensus bbox as this ROI's anchor for text/arrows
    tracker_obj.last_bbox = (mean_p1[0], mean_p1[1],
                                mean_p2[0] - mean_p1[0], mean_p2[1] - mean_p1[1])

    cons_w = float(mean_p2[0] - mean_p1[0])
    cons_h = float(mean_p2[1] - mean_p1[1])

    cx = int(mean_p1[0] + cons_w / 2.0)
    cy = int(mean_p1[1] + cons_h / 2.0)

    cv2.drawMarker(
        frame,
        (cx, cy),
        tracker_obj.colour,
        markerType=cv2.MARKER_CROSS,
        markerSize=14,
        thickness=2,
    )
    # Draw consensus box
    if state.get("show_bbox", True):
        cv2.rectangle(frame, mean_p1, mean_p2, tracker_obj.colour, 2)

    # Now draw your existing overlay items using the consensus bbox as the origin
    x, y, _, _ = tracker_obj.last_bbox

    overlay_colour = tracker_obj.colour

    # Texts (relative offsets)
    for text, (dx, dy) in tracker_obj.texts:
        cv2.putText(frame, text, (int(x) + dx, int(y) + dy + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, overlay_colour, 2)

    # Arrows (relative offsets)
    for (sx, sy), (ex, ey) in tracker_obj.arrows:
        cv2.arrowedLine(frame, (int(x) + sx, int(y) + sy),
                        (int(x) + ex, int(y) + ey),
                        overlay_colour, 2, tipLength=0.3)

def stream_video(
    socketio,
    fps_limit: int,
    target_size: tuple[int, int],
    box_size: int,
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
                with STATE_LOCK:
                    state["paused"] = True
                    break

                # Record resume position after successful read
                state["resume_frame"] = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Resize to tracking/output size
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Initialize tracker on a click
        with STATE_LOCK:
            if state.get("clicked_pt") is not None:
                create_new_tracker(state, "BOX", box_size, frame, socketio)
                state["clicked_pt"] = None  # consume click

        # Update trackers (if initialized)
        with STATE_LOCK:
            tracker_indices = list(range(len(state.get("trackers", []))))

            for i in tracker_indices:
                tracker_obj: Tracker = state["trackers"][i]
                tracker_obj.update(state, frame)

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
        