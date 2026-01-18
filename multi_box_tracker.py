import cv2
import sys

FRAME_RESIZE = 8

ALLOWANCE = 1.0  # Allowance for center point to be near edges
SIZE_ALLOWANCE = 1.5  # Allowance for width comparison


def contains(box_1, box_2, allowance=ALLOWANCE):
    # Check if center of box_2 is inside box_1 with optional allowance
    x1_1, y1_1, w1, h1 = box_1
    x2_1, y2_1, w2, h2 = box_2

    allowance = w1 * allowance  # Convert allowance to pixels based on width of box_1
    
    # Calculate center of box_2
    center_x = x2_1 + w2 / 2
    center_y = y2_1 + h2 / 2
    
    # Check if center is inside box_1 with allowance (margin tolerance)
    if (x1_1 - allowance <= center_x <= x1_1 + w1 + allowance) and (y1_1 - allowance <= center_y <= y1_1 + h1 + allowance):
        return True
    return False


def not_wider(box_1, box_2, size_allowance=SIZE_ALLOWANCE):
    # Check if box_2 is wider than box_1 by a factor of width_allowance
    x1_1, y1_1, w1, h1 = box_1
    x2_1, y2_1, w2, h2 = box_2

    width_allowance = w1 * size_allowance  # Convert allowance to pixels based on width of box_1
    height_allowance = h1 * size_allowance  # Convert allowance to pixels based on height of box_1

    if w2 > width_allowance or h2 > height_allowance:
        return False
    return True


if __name__ == '__main__':
    # Create trackers using legacy module
    tracker1 = cv2.legacy.TrackerKCF_create()
    tracker2 = cv2.legacy.TrackerCSRT_create()
    tracker3 = cv2.legacy.TrackerMedianFlow_create()

    tracker_type1 = "KCF"
    tracker_type2 = "CSRT"
    tracker_type3 = "MEDIANFLOW"
    
    # Read video
    video = cv2.VideoCapture("videos/Echo/echo1.mp4")
    
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    
    # Read first frame
    ok, frame = video.read()
    frame = cv2.resize(frame, None, fx=FRAME_RESIZE, fy=FRAME_RESIZE, interpolation=cv2.INTER_LINEAR)

    if not ok:
        print('Cannot read video file')
        sys.exit()
    
    # Ask user if they want to select ROI manually
    message = "Press 'y' for manual ROI selection, 'n' to use default"
    frame_copy = frame.copy()
    cv2.putText(frame_copy, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.namedWindow('ROI Selector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ROI Selector', 1280, 720)
    cv2.imshow('ROI Selector', frame_copy)

    key = cv2.waitKey(0) & 0xff
    use_manual = (key == ord('y'))

    # Select multiple bboxes
    bboxes = []
    if use_manual:
        print("Select multiple boxes. Press ENTER after selecting each box. Press ESC when done.")
        while True:
            bbox = cv2.selectROI('ROI Selector', frame, False)
            if bbox == (0, 0, 0, 0):  # If no box selected
                break
            bboxes.append(bbox)
            print(f"Box {len(bboxes)}: {bbox}")
    else:
        # Default bboxes (if using defaults)
        bboxes = [(596, 535, 157, 114)]
        print(f"Using default boxes: {bboxes}")
    
    cv2.destroyWindow('ROI Selector')
    
    if not bboxes:
        print("No boxes selected. Exiting.")
        sys.exit()
    
    # Create trackers for each box
    trackers = [[], [], []]  # Three tracker types
    tracker_types = [tracker_type1, tracker_type2, tracker_type3]
    
    for i, bbox in enumerate(bboxes):
        trackers[0].append(cv2.legacy.TrackerKCF_create())
        trackers[1].append(cv2.legacy.TrackerCSRT_create())
        trackers[2].append(cv2.legacy.TrackerMedianFlow_create())
        
        # Initialize trackers with bbox
        trackers[0][i].init(frame, bbox)
        trackers[1][i].init(frame, bbox)
        trackers[2][i].init(frame, bbox)
        
    print(trackers)
    
    # Create tracking window
    cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Tracking', 1280, 720)
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    while True:
        ok, frame = video.read()
        frame = cv2.resize(frame, None, fx=FRAME_RESIZE, fy=FRAME_RESIZE, interpolation=cv2.INTER_LINEAR)

        if not ok:
            break
        
        timer = cv2.getTickCount()
        
        # Track all boxes with all trackers
        results = []  # List of (tracker_type, box_idx, ok, bbox)
        
        for box_idx in range(len(bboxes)):
            for tracker_type_idx in range(3):
                ok_track, bbox = trackers[tracker_type_idx][box_idx].update(frame)
                if ok_track:
                    results.append((tracker_type_idx, box_idx, bbox))
        
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        
        # Draw boxes
        for tracker_type_idx, box_idx, bbox in results:
            x1, y1, w, h = bbox
            p1 = (int(x1), int(y1))
            p2 = (int(x1 + w), int(y1 + h))
            color = colors[box_idx % len(colors)]
            cv2.rectangle(frame, p1, p2, color, 2, 1)
            cv2.putText(frame, f"Box{box_idx+1}-{tracker_types[tracker_type_idx]}", 
                       (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        cv2.putText(frame, f"Tracking {len(bboxes)} boxes", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        
        cv2.imshow("Tracking", frame)
        
        k = cv2.waitKey(22) & 0xff
        if k == 27:  # ESC
            break

    video.release()
    cv2.destroyAllWindows()
