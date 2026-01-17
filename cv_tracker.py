import cv2
import sys

FRAME_RESIZE = 8
BOX_SIZE = 100 

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# store clicked point
clicked_pt = None

# mouse callback for point selection
def on_mouse(event, x, y, flags, param):
    global clicked_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pt = (x, y)

# create a square bbox centered on a point
def make_square_bbox(center, box_size, w, h):
    cx, cy = center
    half = box_size // 2
    x = int(cx - half)
    y = int(cy - half)

    # Clamp to image bounds
    x = max(0, min(x, w - box_size))
    y = max(0, min(y, h - box_size))

    return (x, y, int(box_size), int(box_size))

if __name__ == '__main__' :
 
    # Set up tracker.
    # Instead of MIL, you can also use
 
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[7]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.legacy.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()

    # Read video
    video = cv2.VideoCapture("videos/Echo/echo1.mp4")

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()

    # Resize frame to make it bigger (scale by FRAME_RESIZE)
    frame = cv2.resize(frame, None, fx=FRAME_RESIZE, fy=FRAME_RESIZE, interpolation=cv2.INTER_LINEAR)
 
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    # Click to select a point instead of drawing an ROI
    cv2.namedWindow('ROI Selector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ROI Selector', 1280, 720)
    cv2.setMouseCallback('ROI Selector', on_mouse)

    h, w = frame.shape[:2]

    # wait for user to click a point and confirm
    while True:
        vis = frame.copy()

        if clicked_pt is not None:
            bbox_preview = make_square_bbox(clicked_pt, BOX_SIZE, w, h)
            x, y, bw, bh = bbox_preview
            cv2.drawMarker(
                vis,
                clicked_pt,
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=14,
                thickness=2
            )

        cv2.imshow('ROI Selector', vis)
        k = cv2.waitKey(20) & 0xff

        if k == 27:  # ESC
            cv2.destroyWindow('ROI Selector')
            sys.exit(0)

        if k == 13 and clicked_pt is not None:  # ENTER
            break

    cv2.destroyWindow('ROI Selector')

    # Create a square bounding box around the clicked point 
    bbox = make_square_bbox(clicked_pt, BOX_SIZE, w, h)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    
    # Create a resizable window and set its size
    cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Tracking', 1280, 720)  # Width, Height in pixels
 
    while True:
        # Read a new frame
        ok, frame = video.read()

        # Resize frame to make it bigger (scale by FRAME_RESIZE)
        frame = cv2.resize(frame, None, fx=FRAME_RESIZE, fy=FRAME_RESIZE, interpolation=cv2.INTER_LINEAR)
 
        if not ok:
            break
        
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, bbox = tracker.update(frame)
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
        # Draw bounding box
        if ok:
            x, y, bw, bh = bbox
            cx = int(x + bw / 2)
            cy = int(y + bh / 2)

            # Display as a point/crosshair
            cv2.drawMarker(
                frame,
                (cx, cy),
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=14,
                thickness=2
            )

        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (0,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),2)
 
        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,170,50),2);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,170,50), 2);
 
        # Display result
        cv2.imshow("Tracking", frame)
 
        # Exit if ESC pressed
        k = cv2.waitKey(22) & 0xff
        if k == 27 : break

    cv2.destroyAllWindows()