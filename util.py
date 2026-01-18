import cv2
import base64
import threading

from tracker_fusion import draw_tracked_boxes
from lines_kcf_csrt import draw_lines

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
    def __init__(self, state, tracker_num=None):
        if tracker_num is None:
            self.tracker_num = len(state["trackers"])
        else:
            self.tracker_num = tracker_num

        self.colour = generate_tracker_colour(self.tracker_num)

        self.texts = []
        self.arrows = []
        
        self.tracker_inited = True

    def update(self, state, frame, paused=False):
        # To be implemented in subclasses
        pass

    def add_text(self, text_item):
        return self.texts.append(text_item)
    
    def add_arrow(self, arrow_item):
        return self.arrows.append(arrow_item)
        
class BoxTracker(Tracker):
    def __init__(self, state, box_size, pt, frame, tracker_num=None):
        h, w = frame.shape[:2]

        self.box_size = box_size

        self.bbox0 = make_square_bbox(pt, box_size, w, h)
        self.last_bbox = self.bbox0

        self.fundamental_trackers: list[FundamentalTracker] = []

        # Initialize fundamental trackers
        self.fundamental_trackers.append(FundamentalTracker("KCF", frame, self.bbox0))
        self.fundamental_trackers.append(FundamentalTracker("CSRT", frame, self.bbox0))
        self.fundamental_trackers.append(FundamentalTracker("MEDIANFLOW", frame, self.bbox0))

        super().__init__(state, tracker_num=tracker_num)
        self.update(state, frame, paused=False)

    def update(self, state, frame, paused=None):
        update_box_tracker(state, frame, self, paused)

        # Now draw your existing overlay items using the consensus bbox as the origin
        x, y, _, _ = self.last_bbox

        overlay_colour = self.colour

        # Texts (relative offsets)
        for item in self.texts:
            text, pos = item
            dx, dy = pos["offsets"]
            cv2.putText(frame, text, (int(x) + dx, int(y) + dy + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, overlay_colour, 2)

        # Arrows (relative offsets)
        for start_pos, end_pos in self.arrows:
            sx, sy = start_pos["offsets"]
            ex, ey = end_pos["offsets"]
            cv2.arrowedLine(frame, (int(x) + sx, int(y) + sy),
                            (int(x) + ex, int(y) + ey),
                            overlay_colour, 2, tipLength=0.2)


class LineTracker(Tracker):
    def __init__(self, state, box_size, frame):
        self.points: list[BoxTracker] = []

        super().__init__(state)

    def add_points(self, state, box_size, pts, frame):
        for pt in pts:
            new_tracker = BoxTracker(state, box_size, pt, frame, tracker_num=self.tracker_num)
            self.points.append(new_tracker)

    def update(self, state, frame, paused=None):
        for pt_tracker in self.points:
            pt_tracker.update(state, frame)
        
        # Draw lines between adjacent trackers
        if len(self.points) > 1:
            centers = []
            for pt_tracker in self.points:
                x, y, w, h = pt_tracker.last_bbox
                cx = int(x + w / 2)
                cy = int(y + h / 2)
                centers.append((cx, cy))
            draw_lines(frame, centers, self.colour)
        
        # Draw texts and arrows for each point
        for text, pos in self.texts:
            idx = pos["closest_tracker_index"]
            dx, dy = pos["offsets"]
            if idx < len(self.points):
                px, py, _, _ = self.points[idx].last_bbox
                cv2.putText(frame, text, (int(px) + dx, int(py) + dy + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colour, 2)
        
        for start_pos, end_pos in self.arrows:
            idx = start_pos["closest_tracker_index"]
            sx, sy = start_pos["offsets"]
            ex, ey = end_pos["offsets"]
            if idx < len(self.points):
                px, py, _, _ = self.points[idx].last_bbox
                cv2.arrowedLine(frame, (int(px) + sx, int(py) + sy),
                                (int(px) + ex, int(py) + ey),
                                self.colour, 2, tipLength=0.2)

class PolygonTracker(Tracker):
    def __init__(self, state):
        self.points: list[BoxTracker] = []

        super().__init__(state)

    def add_point(self, state, box_size, pt, frame):
        new_tracker = BoxTracker(state, box_size, pt, frame, tracker_num=self.tracker_num)
        self.points.append(new_tracker)

    def update(self, state, frame, paused=None):
        for pt_tracker in self.points:
            pt_tracker.update(state, frame)
        
        # Draw lines between adjacent trackers
        if len(self.points) > 1:
            centers = []
            for pt_tracker in self.points:
                x, y, w, h = pt_tracker.last_bbox
                cx = int(x + w / 2)
                cy = int(y + h / 2)
                centers.append((cx, cy))
            
            # Draw lines between consecutive centers
            for i in range(len(centers) - 1):
                cv2.line(frame, centers[i], centers[i + 1], self.colour, 2)
            
            # Close the polygon by drawing line from last to first
            cv2.line(frame, centers[-1], centers[0], self.colour, 2)
        
        # Draw texts and arrows for each point
        for text, pos in self.texts:
            idx = pos["closest_tracker_index"]
            dx, dy = pos["offsets"]
            if idx < len(self.points):
                px, py, _, _ = self.points[idx].last_bbox
                cv2.putText(frame, text, (int(px) + dx, int(py) + dy + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colour, 2)
        
        for start_pos, end_pos in self.arrows:
            idx = start_pos["closest_tracker_index"]
            sx, sy = start_pos["offsets"]
            ex, ey = end_pos["offsets"]
            if idx < len(self.points):
                px, py, _, _ = self.points[idx].last_bbox
                cv2.arrowedLine(frame, (int(px) + sx, int(py) + sy),
                                (int(px) + ex, int(py) + ey),
                                self.colour, 2, tipLength=0.2)
    


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

def add_text_to_tracker(tracker_obj, text, offset_info):
    tracker_obj.add_text((text, offset_info))

def add_arrow_to_tracker(tracker_obj, start_point, end_point):
    tracker_obj.add_arrow((start_point, end_point))

def create_new_tracker(state, box_size, frame, socketio):
    tracker_type = state.get("create_new_tracker_type")
    if tracker_type not in ["BOX", "LINE", "POLYGON"]:
        socketio.emit("status", f"Error: Unknown tracker type: {tracker_type}")
        return
        
    with STATE_LOCK:
        pt = state.get("clicked_pt")
        state["clicked_pt"] = None
    
    new_tracker: Tracker | None = None
    if tracker_type == "BOX":
        if pt is None:
            return
        
        new_tracker = BoxTracker(state, box_size, pt, frame)
    elif tracker_type == "LINE":
        new_tracker = LineTracker(state, box_size, frame)
    elif tracker_type == "POLYGON":
        new_tracker = PolygonTracker(state)
    
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

def update_box_tracker(state, frame, tracker_obj: BoxTracker, paused=None):
    if paused is None:
        paused = state.get("paused", False)

    subtrackers = tracker_obj.fundamental_trackers

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
            if state.get("create_new_tracker_type") is not None:
                # We need to create a new tracker
                create_new_tracker(state, box_size, frame, socketio)
                state["create_new_tracker_type"] = None  # consume click

            if state.get("add_point_to_polygon_tracker") is not None:
                idx, pt = state.get("add_point_to_polygon_tracker", (None, None))
                if 0 <= idx < len(state.get("trackers", [])):
                    tracker_obj: Tracker = state["trackers"][idx]
                    if isinstance(tracker_obj, PolygonTracker):
                        tracker_obj.add_point(state, box_size, pt, frame)
                state["add_point_to_polygon_tracker"] = None  # consume

            if state.get("add_points_to_line") is not None:
                idx, pts = state.get("add_points_to_line", (None, []))
                print(f"Adding points to line tracker {idx}: {pts}")
                if 0 <= idx < len(state.get("trackers", [])):
                    tracker_obj: Tracker = state["trackers"][idx]
                    if isinstance(tracker_obj, LineTracker):
                        tracker_obj.add_points(state, box_size, pts, frame)
                state["add_points_to_line"] = None  # consume

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
        