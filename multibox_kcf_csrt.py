from kcf_to_csrt import contains, not_wider, ALLOWANCE, SIZE_ALLOWANCE
from line_tracker import interpolate_line_points, draw_smooth_line
import cv2
import sys
import numpy as np
import itertools

def point_to_bbox(point, box_size):
    """Convert a point to a bounding box centered at that point."""
    x, y = point
    half_size = box_size // 2
    return (x - half_size, y - half_size, box_size, box_size)
tracker_type1 = "KCF"
tracker_type2 = "CSRT"
tracker_type3 = "MEDIANFLOW"
tracker_types = [tracker_type1, tracker_type2, tracker_type3]

resize = 0.8
NICE_x = 748 *resize
NICE_y = 1134 *resize

PATH = "videos/Lapchole/Lapchole1.mp4"

BOXES = False

bbox1 = bbox2 = bbox3 = (0,0,0,0)

BOX_WIDTH = BOX_HEIGHT = 100
BOX_SPACING = 40

def define_boxes(bbox1, bbox2, bbox3, ok1=True, ok2=True, ok3=True, frame=None, drawn_center=(0, 0), boxes = True):
        # Keep track of drawn box centers for line drawing

        
        if ok1:
            # Draw bounding box from tracker 1
            p1_1 = (int(bbox1[0]), int(bbox1[1]))
            p1_2 = (int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3]))
            if boxes:
                cv2.rectangle(frame, p1_1, p1_2, (100, 0, 0), 2, 1)
                # cv2.putText(frame, f"{tracker_types[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 0, 0), 2)

            
            # Calculate center of bbox1
            center1 = (int(bbox1[0] + bbox1[2]/2), int(bbox1[1] + bbox1[3]/2))
            drawn_center = center1

        elif ok2 and contains(bbox1, bbox2) and not_wider(bbox1, bbox2):
            # Draw bounding box from tracker 2
            p2_1 = (int(bbox2[0]), int(bbox2[1]))
            p2_2 = (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))
            if boxes:
                cv2.rectangle(frame, p2_1, p2_2, (0, 100, 0), 2, 1)
                cv2.putText(frame, f"{tracker_types[1]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 0), 2)

            # Calculate center of bbox2
            center2 = (int(bbox2[0] + bbox2[2]/2), int(bbox2[1] + bbox2[3]/2))
            drawn_center = center2





        elif ok3 and contains(bbox1, bbox3) and not_wider(bbox1, bbox3):
            # Draw bounding box from tracker 3
            p3_1 = (int(bbox3[0]), int(bbox3[1]))
            p3_2 = (int(bbox3[0] + bbox3[2]), int(bbox3[1] + bbox3[3]))
            if boxes:
                cv2.rectangle(frame, p3_1, p3_2, (0, 0, 100), 2, 1)
                # cv2.putText(frame, f"{tracker_types[2]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 100), 2)

            
            # Calculate center of bbox3
            center3 = (int(bbox3[0] + bbox3[2]/2), int(bbox3[1] + bbox3[3]/2))
            drawn_center = center3
            

        
        # else:
        #     if boxes:
        #         cv2.putText(frame, "Tracking failure detected", (10, 30), 
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


        
        
        # Display info
        
        # cv2.putText(frame, f"{tracker_type1} + {tracker_type2}", (10, 30), 
        # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 170, 50), 2)


        # cv2.putText(frame, f"Box12, Box23, width12, width23: {contains(bbox1, bbox2), contains(bbox2, bbox3), not_wider(bbox1, bbox2), not_wider(bbox2, bbox3)}", (10, 90), 
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (204, 0, 204), 2)
        

        return drawn_center

def multi_line_select(frame, box_width=50, box_height=50, spacing=30):
    """
    Allow user to trace multiple lines, then create boxes centered at points along each line.
    
    Args:
        frame: Input image
        box_width: Width of boxes to create
        box_height: Height of boxes to create
        spacing: Distance between box centers along each line
    
    Returns:
        List of bounding boxes (x, y, w, h) centered at points along all traced lines
    """
    all_bboxes = []
    all_lines = []
    current_line = []
    drawing = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, current_line, all_lines, all_bboxes
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            current_line = [(x, y)]
            print(f"Started line at ({x}, {y})")
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current_line.append((x, y))
            # Draw preview of all lines and current line
            frame_preview = frame.copy()
            # Draw previous lines
            for line in all_lines:
                cv2.polylines(frame_preview, [np.array(line)], False, (0, 255, 0), 2)
            # Draw current line being drawn
            cv2.polylines(frame_preview, [np.array(current_line)], False, (0, 255, 255), 2)
            cv2.imshow('Multi Line Select', frame_preview)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if len(current_line) >= 1:
                all_lines.append(current_line)
                print(f"Finished line {len(all_lines)} with {len(current_line)} points")
            current_line = []
    
    cv2.namedWindow('Multi Line Select', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Multi Line Select', 1280, 720)
    
    # Draw initial frame with instructions
    frame_display = frame.copy()
    cv2.putText(frame_display, "Draw lines by clicking and dragging.", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame_display, "Press SPACE when done with all lines.", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow('Multi Line Select', frame_display)
    cv2.setMouseCallback('Multi Line Select', mouse_callback)
    
    print("Draw lines by clicking and dragging. Press SPACE when done with all lines.")
    while True:
        key = cv2.waitKey(0) & 0xff
        if key == ord(' '):  # SPACE
            break
    
    cv2.destroyWindow('Multi Line Select')
    
    # Process each line and create boxes
    for line_idx, line_points in enumerate(all_lines):
        all_bboxes.append([])
        if len(line_points) < 2:
            x, y = line_points[0]
            bbox = (int(x - box_width/2), int(y - box_height/2), box_width, box_height)
            all_bboxes[line_idx].append(bbox)

            continue
        
        # Calculate distances along the line
        total_distance = 0
        distances = [0]
        
        for i in range(1, len(line_points)):
            dx = line_points[i][0] - line_points[i-1][0]
            dy = line_points[i][1] - line_points[i-1][1]
            dist = np.sqrt(dx**2 + dy**2)
            total_distance += dist
            distances.append(total_distance)
        
        # Create boxes at regular spacing intervals
        current_distance = 0
        while current_distance <= total_distance:
            # Find the point at current_distance along the line
            idx = 0
            for i, d in enumerate(distances):
                if d <= current_distance:
                    idx = i
                else:
                    break
            
            # Interpolate between line_points[idx] and line_points[idx+1]
            if idx >= len(line_points) - 1:
                point = line_points[-1]
            else:
                segment_dist = current_distance - distances[idx]
                segment_length = distances[idx+1] - distances[idx]
                if segment_length > 0:
                    t = segment_dist / segment_length
                    x = line_points[idx][0] + t * (line_points[idx+1][0] - line_points[idx][0])
                    y = line_points[idx][1] + t * (line_points[idx+1][1] - line_points[idx][1])
                    point = (x, y)
                else:
                    point = line_points[idx]
            
            # Create box centered at this point
            x, y = point
            bbox = (int(x - box_width/2), int(y - box_height/2), box_width, box_height)
            all_bboxes[line_idx].append(bbox)
            
            current_distance += spacing
        
        print(f"Created {len(all_bboxes[line_idx])} boxes from line {line_idx + 1}")
    
    return all_bboxes

def line_select(frame, box_width=50, box_height=50, spacing=30):
    """
    Allow user to trace a line, then create boxes centered at points along that line.
    
    Args:
        frame: Input image
        box_width: Width of boxes to create
        box_height: Height of boxes to create
        spacing: Distance between box centers along the line
    
    Returns:
        List of bounding boxes (x, y, w, h) centered at points along the traced line
    """
    line_points = []
    drawing = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, line_points
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            line_points = [(x, y)]
            print(f"Started line at ({x}, {y})")
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            line_points.append((x, y))
            # Draw preview
            frame_preview = frame.copy()
            cv2.polylines(frame_preview, [np.array(line_points)], False, (0, 255, 255), 2)
            cv2.imshow('Line Select', frame_preview)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            print(f"Finished line with {len(line_points)} points")
    
    import numpy as np
    
    cv2.namedWindow('Line Select', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Line Select', 1280, 720)
    cv2.imshow('Line Select', frame)
    cv2.setMouseCallback('Line Select', mouse_callback)
    
    print("Draw a line by clicking and dragging. Press SPACE when done.")
    while True:
        key = cv2.waitKey(0) & 0xff
        if key == ord(' '):  # SPACE
            break
    
    cv2.destroyWindow('Line Select')
    
    if len(line_points) < 2:
        print("Not enough points to create line")
        return []
    
    # Interpolate boxes along the line at regular spacing intervals
    bboxes = []
    total_distance = 0
    distances = [0]
    
    # Calculate cumulative distances
    for i in range(1, len(line_points)):
        dx = line_points[i][0] - line_points[i-1][0]
        dy = line_points[i][1] - line_points[i-1][1]
        dist = np.sqrt(dx**2 + dy**2)
        total_distance += dist
        distances.append(total_distance)
    
    # Create boxes at regular spacing intervals
    current_distance = 0
    while current_distance <= total_distance:
        # Find the point at current_distance along the line
        idx = 0
        for i, d in enumerate(distances):
            if d <= current_distance:
                idx = i
            else:
                break
        
        # Interpolate between line_points[idx] and line_points[idx+1]
        if idx >= len(line_points) - 1:
            point = line_points[-1]
        else:
            segment_dist = current_distance - distances[idx]
            segment_length = distances[idx+1] - distances[idx]
            if segment_length > 0:
                t = segment_dist / segment_length
                x = line_points[idx][0] + t * (line_points[idx+1][0] - line_points[idx][0])
                y = line_points[idx][1] + t * (line_points[idx+1][1] - line_points[idx][1])
                point = (x, y)
            else:
                point = line_points[idx]
        
        # Create box centered at this point
        x, y = point
        bbox = (int(x - box_width/2), int(y - box_height/2), box_width, box_height)
        bboxes.append(bbox)
        
        current_distance += spacing
    
    print(f"Created {len(bboxes)} boxes along the line")
    return bboxes

def multi_select():
    
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
        bboxes = [(596, 535, 157, 114), (437.0, 645.0, 172.0, 99.0)]
        print(f"Using default boxes: {bboxes}")
    
    cv2.destroyWindow('ROI Selector')
    if not bboxes:
        print("No boxes selected. Exiting.")
        sys.exit()
    
    return bboxes, frame

def initialize_trackers(bboxes_or_points, frame, trackers=[[], [],[]], points=False):
    if points == True:
        for i, point in enumerate(bboxes_or_points):
            bbox = point_to_bbox(point, BOX_WIDTH)
            trackers[0].append(cv2.legacy.TrackerKCF_create())
            trackers[1].append(cv2.legacy.TrackerCSRT_create())
            trackers[2].append(cv2.legacy.TrackerMedianFlow_create())
            
            # Initialize trackers with bbox
            trackers[0][i].init(frame, bbox)
            trackers[1][i].init(frame, bbox)
            trackers[2][i].init(frame, bbox)
            
    
    else:
        for i, bboxes_or_points in enumerate(bboxes_or_points):
            trackers[0].append(cv2.legacy.TrackerKCF_create())
            trackers[1].append(cv2.legacy.TrackerCSRT_create())
            trackers[2].append(cv2.legacy.TrackerMedianFlow_create())
            
            # Initialize trackers with bbox
            trackers[0][i].init(frame, bboxes_or_points)
            trackers[1][i].init(frame, bboxes_or_points)
            trackers[2][i].init(frame, bboxes_or_points)

    return trackers

def init_video(path):
    video = cv2.VideoCapture(path)
    
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    
    # Read first frame
    ok, frame = video.read()
    frame = cv2.resize(frame, None, fx=NICE_x/frame.shape[0], fy=NICE_y/frame.shape[1], interpolation=cv2.INTER_LINEAR)

    return frame, video

def next_frame(frame, video):
    ok, frame = video.read()
    frame = cv2.resize(frame, None, fx=NICE_x/frame.shape[0], fy=NICE_y/frame.shape[1], interpolation=cv2.INTER_LINEAR)


    return ok, frame

def update_lines(trackers, frame, boxes, i, flags, bbox1, bbox2, bbox3):
    points_to_draw = []
    # Update each tracker
    ok1 = ok2 = ok3 = False
    ok1, dummy1 = trackers[0][i].update(frame)
    if ok1:
        bbox1 = dummy1
        flags[i] = True  # Tracker 1 is working


    elif flags[i]:
        ok2, dummy2 = trackers[1][i].update(frame)
    

        if ok2:
            bbox2 = dummy2
        else:
            ok3, dummy3 = trackers[2][i].update(frame)
            if ok3 and flags[i]:
                bbox3 = dummy3
            else:
                flags[i] = False  # Mark this tracker as failed
    

    #cv2.putText(frame, f"ok1: {ok1}, ok2: {ok2}, ok3: {ok3}", (10, 100), 
    #    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (204, 0, 204), 2)
    
    drawn_centre = define_boxes(bbox1, bbox2, bbox3, ok1, ok2, ok3, frame, boxes=boxes)
    
    if drawn_centre != (0, 0):
        points_to_draw.append(drawn_centre)

    return bbox1, bbox2, bbox3, points_to_draw

def active_line_tracking(trackers, frame, count, flags, bbox1, bbox2, bbox3):
    drawn_centres = []
    boxes = BOXES
    if len(line) == 1:
        boxes = True
    

    for i in range(count, count + len(line)): # Updating trackers for all the boxes in the current line
        bbox1, bbox2, bbox3, points_to_draw = update_lines(trackers, frame, boxes, i, flags, bbox1, bbox2, bbox3)
        drawn_centres.extend(points_to_draw)

    count += len(line)
    return drawn_centres, count

def draw_lines(frame, drawn_centres):


    if len(drawn_centres) > 1:
        for i in range(len(drawn_centres) - 1):
            cv2.line(frame, drawn_centres[i], drawn_centres[i+1], (200, 200, 0), 2)

def start_line_tracking(bboxes, frame):
    flat = list(itertools.chain.from_iterable(all_bboxes))


    trackers = initialize_trackers(flat, frame)

    for tracker in trackers:
        print(tracker)
    flags = [True] * len(flat)
    return trackers, flags

if __name__ == '__main__':
    # Initialize video
    frame, video = init_video(PATH)
    print(frame.shape)

    # Let user select multiple lines and create boxes along them

    all_bboxes = [[(323, 487, 100, 100), (330, 448, 100, 100), (336, 411, 100, 100), (336, 371, 100, 100), (336, 331, 100, 100), (361, 316, 100, 100), (400, 314, 100, 100), (438, 321, 100, 100), (475, 330, 100, 100), (513, 335, 100, 100)], [(416, 4, 100, 100), (417, 43, 100, 100), (419, 82, 100, 100), (417, 122, 100, 100), (406, 154, 100, 100)], [(214, 36, 100, 100), (193, 65, 100, 100), (191, 104, 100, 100), (181, 140, 100, 100), (185, 172, 100, 100), (209, 198, 100, 100), (232, 225, 100, 100), (254, 252, 100, 100), (279, 278, 100, 100)]]
    all_bboxes = multi_line_select(frame, box_width=BOX_WIDTH, box_height=BOX_HEIGHT, spacing=BOX_SPACING)
    
    trackers, flags = start_line_tracking(all_bboxes, frame)


    while True:
        ok, frame = next_frame(frame, video)

        if not ok: break

        
        timer = cv2.getTickCount()



        count = 0
        for line in all_bboxes:
            drawn_centres, count = active_line_tracking(trackers, frame, count, flags=flags, bbox1=bbox1, bbox2=bbox2, bbox3=bbox3)
            draw_lines(frame, drawn_centres)



        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        cv2.imshow("Tracking", frame)



        k = cv2.waitKey(22) & 0xff
        if k == 27:  # ESC
            break

    video.release()
    cv2.destroyAllWindows()


"""
The code flow:

Take a set of boxes and a frame -> start_line_tracking -> returns the trackers

For each line, pass those trackers and frame to active_line_tracking -> nested is draw_lines which does the tracker updates

draw_lines updates the drawn_centres list and returns the updated bboxes for next iteration
"""