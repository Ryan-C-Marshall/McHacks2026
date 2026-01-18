import cv2
import sys
import numpy as np
from kcf_to_csrt import contains

FRAME_RESIZE = 0.2
PATH = "videos/POCUS/Vascular.mp4"


def draw_tracked_boxes(frame, trackers_data):
    """
    Draw tracking boxes and compute weighted consensus from multiple trackers.
    """
    
    for t in trackers_data:
        for j in trackers_data:
            if t != j and t['ok'] and j['ok'] and not(contains(t['bbox'], j['bbox'])):
                j['ok'] = False
    
    # Filter successful trackers
    successful = [t for t in trackers_data if t['ok']]
    if not successful:
        cv2.putText(frame, "Tracking failure detected", (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        return frame
    
    # Calculate weights based on which trackers are active
    weights = []
    names = [t['name'] for t in successful]
    
    if 'KCF' in names:
        remaining = 0.30
        for t in successful:
            if t['name'] == 'KCF':
                weights.append(0.70)
            elif t['name'] == 'CSRT':
                weights.append(0.50 * remaining)
            else:
                other_count = sum(1 for n in names if n not in ['KCF', 'CSRT'])
                if 'CSRT' in names:
                    weights.append((0.50 * remaining) / other_count if other_count > 0 else 0)
                else:
                    weights.append(remaining / other_count if other_count > 0 else 0)
    else:
        if 'CSRT' in names:
            for t in successful:
                if t['name'] == 'CSRT':
                    weights.append(0.50)
                else:
                    other_count = sum(1 for n in names if n != 'CSRT')
                    weights.append(0.50 / other_count if other_count > 0 else 0)
        else:
            equal_weight = 1.0 / len(successful)
            weights = [equal_weight] * len(successful)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Collect points
    all_p1 = []
    all_p2 = []
    
    for t, weight in zip(successful, weights):
        bbox = t['bbox']
        color = t['color']
        name = t['name']
        
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        
        cv2.rectangle(frame, p1, p2, color, 1, 1)
        
        label_pos = (p1[0], p1[1] - 5)
        cv2.putText(frame, f"{name} {int(weight*100)}%", label_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        all_p1.append(p1)
        all_p2.append(p2)
    
    # Draw consensus box
    if len(successful) > 0:
        mean_p1 = (
            int(sum(p[0] * w for p, w in zip(all_p1, weights))),
            int(sum(p[1] * w for p, w in zip(all_p1, weights)))
        )
        mean_p2 = (
            int(sum(p[0] * w for p, w in zip(all_p2, weights))),
            int(sum(p[1] * w for p, w in zip(all_p2, weights)))
        )
        
        cv2.rectangle(frame, mean_p1, mean_p2, (0, 255, 255), 1, 1)
        cv2.putText(frame, "Consensus", (mean_p1[0], mean_p1[1] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    return frame


def rotate_image(image, angle, center=None):
    """
    Rotate image around center by angle (degrees).
    """
    h, w = image.shape[:2]
    
    if center is None:
        center = (w // 2, h // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate the image
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    
    return rotated, M


def rotate_bbox(bbox, rotation_matrix):
    """
    Rotate a bounding box using rotation matrix.
    Returns new axis-aligned bounding box after rotation.
    """
    x, y, w, h = bbox
    
    # Get four corners of bbox
    corners = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ], dtype=np.float32)
    
    # Add ones for affine transformation
    ones = np.ones((4, 1), dtype=np.float32)
    corners_homo = np.hstack([corners, ones])
    
    # Apply rotation
    rotated_corners = rotation_matrix.dot(corners_homo.T).T
    
    # Get new axis-aligned bounding box
    x_coords = rotated_corners[:, 0]
    y_coords = rotated_corners[:, 1]
    
    new_x = int(np.min(x_coords))
    new_y = int(np.min(y_coords))
    new_w = int(np.max(x_coords) - new_x)
    new_h = int(np.max(y_coords) - new_y)
    
    return (new_x, new_y, new_w, new_h)


def draw_rotated_box(frame, bbox, angle, color=(0, 255, 255), thickness=2):
    """
    Draw a rotated rectangle on the frame.
    """
    x, y, w, h = bbox
    center = (x + w // 2, y + h // 2)
    
    rect = (center, (w, h), angle)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    cv2.drawContours(frame, [box], 0, color, thickness)


def select_rotated_roi(frame):
    """
    Interactive rotated rectangle selection.
    """
    state = {
        'drawing': False,
        'start_point': None,
        'current_point': None,
        'initial_y': None,
        'angle': 0,
        'finished': False,
        'cancelled': False
    }
    
    clone = frame.copy()
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state['drawing'] = True
            state['start_point'] = (x, y)
            state['current_point'] = (x, y)
            state['initial_y'] = y
            state['angle'] = 0
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if state['drawing']:
                state['current_point'] = (x, y)
                y_diff = state['initial_y'] - y
                state['angle'] = y_diff * 0.5
            
            temp = clone.copy()
            draw_rotated_rectangle(temp, state)
            cv2.imshow('ROI Selector', temp)
            
        elif event == cv2.EVENT_LBUTTONUP:
            state['drawing'] = False
    
    def draw_rotated_rectangle(img, state):
        cv2.putText(img, "Drag: resize | Move up/down: rotate | ENTER: confirm | ESC: cancel", 
                   (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if state['start_point'] is None or state['current_point'] is None:
            return
        
        x1, y1 = state['start_point']
        x2, y2 = state['current_point']
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        if width < 5 or height < 5:
            cv2.putText(img, "Draw a larger rectangle", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return
        
        rect = ((center_x, center_y), (width, height), state['angle'])
        
        try:
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            
            cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
            cv2.circle(img, (int(center_x), int(center_y)), 3, (0, 0, 255), -1)
            
            angle_text = f"Angle: {int(state['angle'])}° | Size: {int(width)}x{int(height)}"
            cv2.putText(img, angle_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error: {e}")
    
    cv2.namedWindow('ROI Selector')
    cv2.setMouseCallback('ROI Selector', mouse_callback)
    
    temp = clone.copy()
    draw_rotated_rectangle(temp, state)
    cv2.imshow('ROI Selector', temp)
    
    print("Instructions:")
    print("- Click and drag to set size")
    print("- Move mouse UP/DOWN while dragging to rotate")
    print("- Press SPACE or ENTER to confirm")
    print("- Press ESC to cancel")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13 or key == 32:
            state['finished'] = True
            break
        elif key == 27:
            state['cancelled'] = True
            break
    
    cv2.destroyWindow('ROI Selector')
    
    if state['cancelled'] or state['start_point'] is None or state['current_point'] is None:
        return None, None
    
    x1, y1 = state['start_point']
    x2, y2 = state['current_point']
    
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    
    if width < 5 or height < 5:
        return None, None
    
    bbox = (
        int(center_x - width / 2),
        int(center_y - height / 2),
        int(width),
        int(height)
    )
    
    rotated_rect = ((center_x, center_y), (width, height), state['angle'])
    
    return bbox, rotated_rect


if __name__ == '__main__':
    # Read video
    video = cv2.VideoCapture(PATH)
    
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    
    # Read first frame
    ok, frame = video.read()
    frame = cv2.resize(frame, None, fx=FRAME_RESIZE, fy=FRAME_RESIZE, interpolation=cv2.INTER_LINEAR)

    if not ok:
        print('Cannot read video file')
        sys.exit()
    
    # Select rotated ROI
    cv2.namedWindow('ROI Selector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ROI Selector', 1280, 720)
    
    bbox, rotated_rect = select_rotated_roi(frame)
    
    if bbox is None:
        print("No valid ROI selected")
        sys.exit()
    
    # Store the angle
    rotation_angle = rotated_rect[2]
    print(f"Tracking with rotation: {rotation_angle:.1f}°")
    
    # Rotate the frame to align ROI
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    rotated_frame, rotation_matrix = rotate_image(frame, -rotation_angle, center)
    
    # Transform bbox to rotated frame coordinates
    rotated_bbox = rotate_bbox(bbox, rotation_matrix)
    
    # Initialize trackers on rotated frame
    tracker1 = cv2.legacy.TrackerKCF_create()
    tracker2 = cv2.legacy.TrackerCSRT_create()
    tracker3 = cv2.legacy.TrackerMedianFlow_create()
    
    tracker1.init(rotated_frame, rotated_bbox)
    tracker2.init(rotated_frame, rotated_bbox)
    tracker3.init(rotated_frame, rotated_bbox)
    
    # Create tracking window
    cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Tracking', 1280, 720)

    # Main tracking loop
    while True:
        ok, frame = video.read()
        if not ok:
            break
            
        frame = cv2.resize(frame, None, fx=FRAME_RESIZE, fy=FRAME_RESIZE, interpolation=cv2.INTER_LINEAR)
        
        # Rotate frame for tracking
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        rotated_frame, rotation_matrix = rotate_image(frame, -rotation_angle, center)
        
        timer = cv2.getTickCount()
        
        # Update trackers on rotated frame
        ok1, bbox1 = tracker1.update(rotated_frame)
        ok2, bbox2 = tracker2.update(rotated_frame)
        ok3, bbox3 = tracker3.update(rotated_frame)
        
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        
        # Calculate inverse rotation to get back to original frame
        inv_rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        
        # Transform bboxes back to original frame
        if ok1:
            bbox1_original = rotate_bbox(bbox1, inv_rotation_matrix)
        if ok2:
            bbox2_original = rotate_bbox(bbox2, inv_rotation_matrix)
        if ok3:
            bbox3_original = rotate_bbox(bbox3, inv_rotation_matrix)
        
        # Draw rotated rectangles on original frame
        if ok1:
            draw_rotated_box(frame, bbox1_original, rotation_angle, (255, 0, 0), 1)
        if ok2:
            draw_rotated_box(frame, bbox2_original, rotation_angle, (0, 255, 0), 1)
        if ok3:
            draw_rotated_box(frame, bbox3_original, rotation_angle, (0, 0, 255), 1)
        
        # Draw consensus if available
        if ok1 and ok2:
            # Simple average for demo (you can use weighted)
            x1, y1, w1, h1 = bbox1_original
            x2, y2, w2, h2 = bbox2_original
            consensus_bbox = (
                int((x1 + x2) / 2),
                int((y1 + y2) / 2),
                int((w1 + w2) / 2),
                int((h1 + h2) / 2)
            )
            draw_rotated_box(frame, consensus_bbox, rotation_angle, (0, 255, 255), 2)
        
        # Display info
        active_count = sum([ok1, ok2, ok3])
        cv2.putText(frame, f"Trackers: {active_count}/3 | Angle: {int(rotation_angle)}°", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 170, 50), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 170, 50), 2)
        
        cv2.imshow("Tracking", frame)
        
        k = cv2.waitKey(22) & 0xff
        if k == 27:
            break
    
    video.release()
    cv2.destroyAllWindows()