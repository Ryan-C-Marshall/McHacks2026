import cv2
import sys
import numpy as np
from kcf_to_csrt import contains

from multibox_kcf_csrt import initialize_trackers

FRAME_RESIZE = 12
PATH = "videos/Echo/echo3.mp4"
BOX_SIZE = 100  # Size of tracking box for each vertex


def draw_tracked_boxes(frame, trackers_data):
    """
    Draw tracking boxes and compute weighted consensus from multiple trackers.
    
    Returns:
        frame: Frame with drawn boxes
        consensus_center: Center point of consensus box, or None if tracking failed
    """
    
    for t in trackers_data:
        for j in trackers_data:
            if t != j and t['ok'] and j['ok'] and not(contains(t['bbox'], j['bbox'])):
                j['ok'] = False
    
    # Filter successful trackers
    successful = [t for t in trackers_data if t['ok']]
    if not successful:
        return frame, None
    
    # Calculate weights based on which trackers are active
    weights = []
    names = [t['name'] for t in successful]
    
    if 'KCF' in names:
        remaining = 0.3
        for t in successful:
            if t['name'] == 'KCF':
                weights.append(0.7)
            elif t['name'] == 'CSRT':
                weights.append(0.75 * remaining)
            else:
                other_count = sum(1 for n in names if n not in ['KCF', 'CSRT'])
                if 'CSRT' in names:
                    weights.append((0.25 * remaining) / other_count if other_count > 0 else 0)
                else:
                    weights.append(remaining / other_count if other_count > 0 else 0)
    else:
        if 'CSRT' in names:
            for t in successful:
                if t['name'] == 'CSRT':
                    weights.append(0.75)
                else:
                    other_count = sum(1 for n in names if n != 'CSRT')
                    weights.append(0.25 / other_count if other_count > 0 else 0)
        else:
            equal_weight = 1.0 / len(successful)
            weights = [equal_weight] * len(successful)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Draw individual tracker boxes and collect points
    all_p1 = []
    all_p2 = []
    
    for t, weight in zip(successful, weights):
        bbox = t['bbox']
        color = t['color']
        name = t['name']
        
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        
        #cv2.rectangle(frame, p1, p2, color, 1, 1)
        
        label_pos = (p1[0], p1[1] - 5)
        #cv2.putText(frame, f"{name} {int(weight*100)}%", label_pos, 
                   #cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        all_p1.append(p1)
        all_p2.append(p2)
    
    # Calculate consensus center
    consensus_center = None
    if len(successful) > 0:
        mean_p1 = (
            int(sum(p[0] * w for p, w in zip(all_p1, weights))),
            int(sum(p[1] * w for p, w in zip(all_p1, weights)))
        )
        mean_p2 = (
            int(sum(p[0] * w for p, w in zip(all_p2, weights))),
            int(sum(p[1] * w for p, w in zip(all_p2, weights)))
        )
        
        # Draw consensus box in cyan
        cv2.rectangle(frame, mean_p1, mean_p2, (0, 255, 255), 2, 1)
        
        # Calculate center of consensus box
        consensus_center = (
            (mean_p1[0] + mean_p2[0]) // 2,
            (mean_p1[1] + mean_p2[1]) // 2
        )
    
    return frame, consensus_center


def select_polygon_roi(frame):
    """Interactive polygon selection by clicking points."""
    state = {
        'points': [],
        'finished': False,
        'cancelled': False
    }
    
    clone = frame.copy()
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state['points'].append((x, y))
            temp = clone.copy()
            draw_polygon(temp, state['points'])
            cv2.imshow('Polygon Selector', temp)
            
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            state['finished'] = True
            
        elif event == cv2.EVENT_MOUSEMOVE:
            temp = clone.copy()
            draw_polygon(temp, state['points'], mouse_pos=(x, y))
            cv2.imshow('Polygon Selector', temp)
    
    def draw_polygon(img, points, mouse_pos=None):
        instruction_text = "Click: add point | Double-click/ENTER: finish | C: clear | ESC: cancel"
        cv2.putText(img, instruction_text, (10, img.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if len(points) == 0:
            cv2.putText(img, "Click to add points for your polygon", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return
        
        for i, point in enumerate(points):
            cv2.circle(img, point, 4, (0, 255, 0), -1)
            cv2.putText(img, str(i+1), (point[0]+8, point[1]-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i+1], (0, 255, 0), 2)
        
        if len(points) >= 2:
            cv2.line(img, points[-1], points[0], (0, 255, 255), 2)
        
        if mouse_pos is not None and len(points) > 0:
            cv2.line(img, points[-1], mouse_pos, (255, 0, 255), 1)
        
        if len(points) >= 3:
            overlay = img.copy()
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
        
        cv2.putText(img, f"Points: {len(points)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.namedWindow('Polygon Selector')
    cv2.setMouseCallback('Polygon Selector', mouse_callback)
    
    temp = clone.copy()
    draw_polygon(temp, state['points'])
    cv2.imshow('Polygon Selector', temp)
    
    print("Instructions:")
    print("- Click to add points to create your polygon")
    print("- The polygon will automatically close (yellow line from last to first point)")
    print("- Double-click or press ENTER when done")
    print("- Press 'C' to clear all points and restart")
    print("- Press ESC to cancel")
    print("- Need at least 3 points to create a valid polygon")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:
            state['finished'] = True
            break
        elif key == 27:
            state['cancelled'] = True
            break
        elif key == ord('c') or key == ord('C'):
            state['points'] = []
            temp = clone.copy()
            draw_polygon(temp, state['points'])
            cv2.imshow('Polygon Selector', temp)
        
        if state['finished']:
            break
    
    cv2.destroyWindow('Polygon Selector')
    
    if state['cancelled'] or len(state['points']) < 3:
        if not state['cancelled']:
            print("Need at least 3 points to create a polygon")
        return None
    
    print(f"Created polygon with {len(state['points'])} points")
    return state['points']


def draw_polygon_on_frame(frame, points, color=(0, 255, 255), thickness=2):
    """Draw a polygon on the frame."""
    if points is None or len(points) < 3:
        return
    
    pts = np.array(points, dtype=np.int32)
    
    # Draw filled polygon semi-transparent
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Draw polygon outline
    cv2.polylines(frame, [pts], True, color, thickness)
    
    # Draw vertices as squares instead of circles
    half_size = 4
    for i, point in enumerate(points):
        x, y = point
        top_left = (x - half_size, y - half_size)
        bottom_right = (x + half_size, y + half_size)
        #cv2.rectangle(frame, top_left, bottom_right, color, -1)
        #cv2.putText(frame, str(i+1), (x+8, y-8), 
                   #cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
def point_to_bbox(point, box_size):
    """Convert a point to a bounding box centered at that point."""
    x, y = point
    half_size = box_size // 2
    return (x - half_size, y - half_size, box_size, box_size)

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
    
    # Select polygon ROI
    cv2.namedWindow('Polygon Selector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Polygon Selector', 1280, 720)
    
    polygon_points = select_polygon_roi(frame)
    
    if polygon_points is None:
        print("No valid polygon selected")
        sys.exit()
    
    # Show final polygon
    # preview = frame.copy()
    # draw_polygon_on_frame(preview, polygon_points)
    # cv2.imshow('Final Polygon', preview)
    # print(f"Preview - Press any key to start tracking... (BOX_SIZE = {BOX_SIZE}px)")
    # cv2.waitKey(1500)
    # cv2.destroyWindow('Final Polygon')
    
    # Create trackers for each vertex
    num_vertices = len(polygon_points)
    vertex_trackers = []
    
    for i, point in enumerate(polygon_points):
        # Create bbox around each vertex point
        x, y = point
        bbox = point_to_bbox(point, BOX_SIZE)
        
        # Create 3 trackers per vertex
        tracker_kcf = cv2.legacy.TrackerKCF_create()
        tracker_csrt = cv2.legacy.TrackerCSRT_create()
        tracker_mf = cv2.legacy.TrackerBoosting_create()
        
        tracker_kcf.init(frame, bbox)
        tracker_csrt.init(frame, bbox)
        tracker_mf.init(frame, bbox)

        
        vertex_trackers.append({
            'vertex_id': i,
            'trackers': [
                {'tracker': tracker_kcf, 'name': 'KCF', 'color': (255, 0, 0)},
                {'tracker': tracker_csrt, 'name': 'CSRT', 'color': (0, 255, 0)},
                {'tracker': tracker_mf, 'name': 'MEDIANFLOW', 'color': (0, 0, 255)}
            ],
            'last_center': point
        })
    

    print(f"Initialized {num_vertices} vertex trackers with {num_vertices * 3} total trackers")
    print(vertex_trackers)
    
    # Create tracking window
    cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Tracking', 1280, 720)

    # Main tracking loop
    frame_count = 0
    while True:
        ok, frame = video.read()
        if not ok:
            break
            
        frame = cv2.resize(frame, None, fx=FRAME_RESIZE, fy=FRAME_RESIZE, interpolation=cv2.INTER_LINEAR)
        frame_count += 1
        
        timer = cv2.getTickCount()
        
        # Track each vertex
        current_polygon = []
        vertices_tracked = 0
        
        for vertex_data in vertex_trackers:
            # Update all trackers for this vertex
            trackers_data = []
            for t_info in vertex_data['trackers']:
                ok, bbox = t_info['tracker'].update(frame)
                trackers_data.append({
                    'ok': ok,
                    'bbox': bbox,
                    'color': t_info['color'],
                    'name': t_info['name']
                })
            
            # Get consensus position for this vertex (draws boxes on frame)
            frame, consensus_center = draw_tracked_boxes(frame, trackers_data)
            
            if consensus_center is not None:
                current_polygon.append(consensus_center)
                vertex_data['last_center'] = consensus_center
                vertices_tracked += 1
                
                # Draw center point as a small square
                x, y = consensus_center
                cv2.rectangle(frame, (x-2, y-2), (x+2, y+2), (0, 255, 255), -1)
                # Label vertex number
                cv2.putText(frame, f"V{vertex_data['vertex_id']+1}", (x+6, y-6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            else:
                # Use last known position
                current_polygon.append(vertex_data['last_center'])
                x, y = vertex_data['last_center']
                # Draw with different color to indicate it's using last known position
                cv2.rectangle(frame, (x-2, y-2), (x+2, y+2), (0, 0, 255), -1)
                cv2.putText(frame, f"V{vertex_data['vertex_id']+1}?", (x+6, y-6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        
        # Draw the polygon if we have enough valid vertices
        if len(current_polygon) >= 3:
            draw_polygon_on_frame(frame, current_polygon, color=(0, 255, 255), thickness=2)
            tracking_status = "TRACKING"
            status_color = (0, 255, 0)
        else:
            tracking_status = "LOST"
            status_color = (0, 0, 255)
        
        # Display info
        cv2.putText(frame, f"Status: {tracking_status} | Vertices: {vertices_tracked}/{num_vertices} | Box Size: {BOX_SIZE}px", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, f"FPS: {int(fps)} | Frame: {frame_count}", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 170, 50), 2)
        
        cv2.imshow("Tracking", frame)
        
        k = cv2.waitKey(22) & 0xff
        if k == 27:  # ESC
            break
    
    video.release()
    cv2.destroyAllWindows()