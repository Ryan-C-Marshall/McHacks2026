import cv2
import sys
from kcf_to_csrt import contains

FRAME_RESIZE = 0.2
PATH = "videos/Microscopy2.py"

def draw_tracked_circles(frame, trackers_data, initial_radius, initial_bbox_size, label="Consensus"):
    """
    Draw tracking circles and compute weighted consensus from multiple trackers.
    Adapts circle radius based on zoom level (bbox size changes).
    
    Weighting scheme:
    - KCF always gets 80% when available
    - CSRT gets 75% of remaining weight when available
    - Other trackers split the rest
    
    Args:
        frame: Current video frame
        trackers_data: List of dicts with keys: 'ok', 'bbox', 'color', 'name'
        initial_radius: Initial circle radius from user selection
        initial_bbox_size: Initial bbox size (width, height) for scale reference
        label: Custom label for the consensus circle
        
    Returns:
        frame: Frame with drawn circles
        out_of_bounds: Boolean indicating if consensus circle is out of frame
    """
    
    frame_height, frame_width = frame.shape[:2]
    border_margin = 10  # Pixels from edge to consider "touching border"
    
    for t in trackers_data:
        for j in trackers_data:
            if t != j and t['ok'] and j['ok'] and not(contains(t['bbox'], j['bbox'])):
                j['ok'] = False
    
    # Filter successful trackers
    successful = [t for t in trackers_data if t['ok']]
    if not successful:
        # No tracking success at all
        cv2.putText(frame, "Tracking failure detected", (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        return frame
    
    # Calculate weights based on which trackers are active
    weights = []
    names = [t['name'] for t in successful]
    
    if 'KCF' in names:
        # KCF gets 80%, distribute remaining 20%
        remaining = 0.20
        for t in successful:
            if t['name'] == 'KCF':
                weights.append(0.80)
            elif t['name'] == 'CSRT':
                # CSRT gets 75% of the remaining
                weights.append(0.75 * remaining)
            else:
                # Others split what's left
                other_count = sum(1 for n in names if n not in ['KCF', 'CSRT'])
                if 'CSRT' in names:
                    weights.append((0.25 * remaining) / other_count if other_count > 0 else 0)
                else:
                    weights.append(remaining / other_count if other_count > 0 else 0)
    else:
        # KCF not available
        if 'CSRT' in names:
            # CSRT gets 75%, others split 25%
            for t in successful:
                if t['name'] == 'CSRT':
                    weights.append(0.75)
                else:
                    other_count = sum(1 for n in names if n != 'CSRT')
                    weights.append(0.25 / other_count if other_count > 0 else 0)
        else:
            # Neither KCF nor CSRT available, equal weights
            equal_weight = 1.0 / len(successful)
            weights = [equal_weight] * len(successful)
    
    # Normalize weights to ensure they sum to 1.0
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Draw individual tracker circles and collect centers
    all_centers = []
    all_radii = []
    
    for t, weight in zip(successful, weights):
        bbox = t['bbox']
        color = t['color']
        name = t['name']
        
        # Calculate center from bounding box
        center_x = int(bbox[0] + bbox[2] / 2)
        center_y = int(bbox[1] + bbox[3] / 2)
        center = (center_x, center_y)
        
        # Check if this tracker's center is out of bounds
        touching_left = center_x <= border_margin
        touching_right = center_x >= frame_width - border_margin
        touching_top = center_y <= border_margin
        touching_bottom = center_y >= frame_height - border_margin
        
        tracker_out_of_bounds = touching_left or touching_right or touching_top or touching_bottom
        
        if tracker_out_of_bounds:
            # Skip this tracker - don't draw it or include it in consensus
            continue
        
        # Adapt radius based on bbox size change (zoom adaptation)
        current_bbox_size = (bbox[2] + bbox[3]) / 2  # Average of width and height
        initial_bbox_avg = (initial_bbox_size[0] + initial_bbox_size[1]) / 2
        scale_factor = current_bbox_size / initial_bbox_avg
        adapted_radius = int(initial_radius * scale_factor)
        
        # Draw circle with adapted radius
        cv2.circle(frame, center, adapted_radius, color, 2)
        
        # Label each circle with tracker name and weight
        label_pos = (center_x - 30, center_y - adapted_radius - 10)
        cv2.putText(frame, f"{name} {int(weight*100)}%", label_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        all_centers.append(center)
        all_radii.append(adapted_radius)
    
    # If all trackers are out of bounds, show warning
    if len(all_centers) == 0:
        cv2.putText(frame, f"{label} - OUT OF FRAME", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(frame, "Feature has left the visible area", (50, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame, True
    
    # Draw consensus circle if we have trackers still in bounds
    if len(all_centers) > 0:
        # Recalculate weights for remaining trackers
        if len(all_centers) < len(successful):
            # Some trackers were filtered out, need to renormalize weights
            remaining_weights = weights[:len(all_centers)]
            total = sum(remaining_weights)
            remaining_weights = [w / total for w in remaining_weights]
        else:
            remaining_weights = weights
        
        # Weighted average of all centers
        consensus_x = int(sum(c[0] * w for c, w in zip(all_centers, remaining_weights)))
        consensus_y = int(sum(c[1] * w for c, w in zip(all_centers, remaining_weights)))
        consensus_center = (consensus_x, consensus_y)
        
        # Weighted average of radii
        consensus_radius = int(sum(r * w for r, w in zip(all_radii, remaining_weights)))
        
        # Final check: is consensus center also out of bounds?
        touching_left = consensus_x <= border_margin
        touching_right = consensus_x >= frame_width - border_margin
        touching_top = consensus_y <= border_margin
        touching_bottom = consensus_y >= frame_height - border_margin
        
        out_of_bounds = touching_left or touching_right or touching_top or touching_bottom
        
        if out_of_bounds:
            # Don't draw the circle, show warning message instead
            cv2.putText(frame, f"{label} - OUT OF FRAME", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(frame, "Feature has left the visible area", (50, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame, True
        else:
            # Draw consensus circle in cyan, thicker
            cv2.circle(frame, consensus_center, consensus_radius, (0, 255, 255), 3)
            
            # Draw center point
            cv2.circle(frame, consensus_center, 3, (0, 255, 255), -1)
            
            # Label
            cv2.putText(frame, label, (consensus_x - 30, consensus_y - consensus_radius - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            return frame, False
    
    return frame, False


if __name__ == '__main__':
    # Create trackers using legacy module
    tracker1 = cv2.legacy.TrackerKCF_create()
    tracker2 = cv2.legacy.TrackerCSRT_create()
    tracker3 = cv2.legacy.TrackerMedianFlow_create()

    tracker_type1 = "KCF"
    tracker_type2 = "CSRT"
    tracker_type3 = "MedianFlow"
    
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
    
    # Interactive circle drawing
    print("Instructions:")
    print("- Click and hold to define the center")
    print("- Drag to expand the circle to desired radius")
    print("- Release to confirm the circle")
    print("- Type to enter a label for the annotation")
    print("- Press ENTER to start tracking")
    print("- Press ESC to exit")
    
    state = {
        'center': None,
        'radius': 0,
        'drawing': False,
        'display_frame': frame.copy(),
        'label': ''
    }
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state['drawing'] = True
            state['center'] = (x, y)
            state['radius'] = 0
            state['display_frame'] = frame.copy()
            cv2.circle(state['display_frame'], (x, y), 3, (0, 255, 255), -1)
            # Show current label if any
            if state['label']:
                cv2.putText(state['display_frame'], f"Label: {state['label']}", 
                           (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('Circle Selector', state['display_frame'])
            
        elif event == cv2.EVENT_MOUSEMOVE and state['drawing']:
            if state['center'] is not None:
                # Calculate radius from center to current mouse position
                dx = x - state['center'][0]
                dy = y - state['center'][1]
                state['radius'] = int((dx**2 + dy**2)**0.5)
                
                # Redraw frame with circle
                state['display_frame'] = frame.copy()
                cv2.circle(state['display_frame'], state['center'], state['radius'], (0, 255, 255), 2)
                cv2.circle(state['display_frame'], state['center'], 3, (0, 255, 255), -1)
                
                # Show radius value
                cv2.putText(state['display_frame'], f"Radius: {state['radius']}px", 
                           (state['center'][0] + 10, state['center'][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Show current label if any
                if state['label']:
                    cv2.putText(state['display_frame'], f"Label: {state['label']}", 
                               (10, frame.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow('Circle Selector', state['display_frame'])
                
        elif event == cv2.EVENT_LBUTTONUP:
            state['drawing'] = False
            if state['radius'] < 10:  # Minimum radius
                state['radius'] = 10
            state['display_frame'] = frame.copy()
            cv2.circle(state['display_frame'], state['center'], state['radius'], (0, 255, 255), 2)
            cv2.circle(state['display_frame'], state['center'], 3, (0, 255, 255), -1)
            
            # Show instruction to type label
            cv2.putText(state['display_frame'], "Type label and press ENTER", 
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show current label
            if state['label']:
                cv2.putText(state['display_frame'], f"Label: {state['label']}", 
                           (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Circle Selector', state['display_frame'])
    
    cv2.namedWindow('Circle Selector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Circle Selector', 1280, 720)
    cv2.setMouseCallback('Circle Selector', mouse_callback)
    cv2.imshow('Circle Selector', state['display_frame'])
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        if key == 13 and state['center'] is not None and state['radius'] > 0:  # ENTER
            break
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            sys.exit()
        elif key == 8 and len(state['label']) > 0:  # BACKSPACE
            state['label'] = state['label'][:-1]
            # Redraw
            state['display_frame'] = frame.copy()
            if state['center'] and state['radius'] > 0:
                cv2.circle(state['display_frame'], state['center'], state['radius'], (0, 255, 255), 2)
                cv2.circle(state['display_frame'], state['center'], 3, (0, 255, 255), -1)
                cv2.putText(state['display_frame'], "Type label and press ENTER", 
                           (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if state['label']:
                cv2.putText(state['display_frame'], f"Label: {state['label']}", 
                           (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('Circle Selector', state['display_frame'])
        elif 32 <= key <= 126:  # Printable ASCII characters
            state['label'] += chr(key)
            # Redraw
            state['display_frame'] = frame.copy()
            if state['center'] and state['radius'] > 0:
                cv2.circle(state['display_frame'], state['center'], state['radius'], (0, 255, 255), 2)
                cv2.circle(state['display_frame'], state['center'], 3, (0, 255, 255), -1)
                cv2.putText(state['display_frame'], "Type label and press ENTER", 
                           (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(state['display_frame'], f"Label: {state['label']}", 
                       (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('Circle Selector', state['display_frame'])
    
    cv2.destroyWindow('Circle Selector')
    
    if state['center'] is None or state['radius'] == 0:
        print("No circle selected")
        sys.exit()
    
    CIRCLE_RADIUS = state['radius']
    CIRCLE_LABEL = state['label'] if state['label'] else "Tracked Object"
    print(f"Circle defined with radius: {CIRCLE_RADIUS}px, label: '{CIRCLE_LABEL}'")
    
    # Convert circle to bounding box for tracker initialization
    cx, cy = state['center']
    bbox = (
        cx - CIRCLE_RADIUS,
        cy - CIRCLE_RADIUS,
        2 * CIRCLE_RADIUS,
        2 * CIRCLE_RADIUS
    )
    
    # Store initial bbox size for zoom adaptation
    initial_bbox_size = (2 * CIRCLE_RADIUS, 2 * CIRCLE_RADIUS)
    
    # Initialize all trackers with bbox from circle
    ok1 = tracker1.init(frame, bbox)
    ok2 = tracker2.init(frame, bbox)
    ok3 = tracker3.init(frame, bbox)
    
    # Create tracking window
    cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Tracking', 1280, 720)

    # Main tracking loop
    while True:
        ok, frame = video.read()
        if not ok:
            break
            
        frame = cv2.resize(frame, None, fx=FRAME_RESIZE, fy=FRAME_RESIZE, interpolation=cv2.INTER_LINEAR)
        
        timer = cv2.getTickCount()
        
        # Update all trackers
        ok1, bbox1 = tracker1.update(frame)
        ok2, bbox2 = tracker2.update(frame)
        ok3, bbox3 = tracker3.update(frame)
        
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        
        # Define tracker data as list of dictionaries
        trackers_data = [
            {'ok': ok1, 'bbox': bbox1, 'color': (255, 0, 0), 'name': 'KCF'},
            {'ok': ok2, 'bbox': bbox2, 'color': (0, 255, 0), 'name': 'CSRT'},
            {'ok': ok3, 'bbox': bbox3, 'color': (0, 0, 255), 'name': 'MEDIANFLOW'}
        ]
        
        # Draw all circles and weighted consensus
        frame, out_of_bounds = draw_tracked_circles(frame, trackers_data, CIRCLE_RADIUS, initial_bbox_size, CIRCLE_LABEL)
        
        # Display info
        active_count = sum([ok1, ok2, ok3])
        status_color = (50, 170, 50) if not out_of_bounds else (0, 0, 255)
        status_text = "IN FRAME" if not out_of_bounds else "OUT OF FRAME"
        
        cv2.putText(frame, f"Trackers: {active_count}/3 active | Radius: {CIRCLE_RADIUS}px | Status: {status_text}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"FPS : {int(fps)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 170, 50), 2)
        
        cv2.imshow("Tracking", frame)
        
        k = cv2.waitKey(22) & 0xff
        if k == 27:  # ESC
            break
    
    video.release()
    cv2.destroyAllWindows()