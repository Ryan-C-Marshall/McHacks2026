import cv2
import sys
import numpy as np
from scipy.interpolate import splprep, splev
from kcf_to_csrt import contains

FRAME_RESIZE = 1
NUM_POINTS = 3
BOX_SIZE = 100

'''
/////////////////////////////////////////////
#1 DRAW BOXES
/////////////////////////////////////////////
'''

def draw_tracked_boxes(frame, trackers_data):
    """
    Draw tracking boxes and compute weighted consensus from multiple trackers.
    
    Weighting scheme:
    - KCF always gets 70% when available
    - CSRT gets 60% of remaining weight when available
    - Other trackers split the rest
    
    Args:
        frame: Current video frame
        trackers_data: List of dicts with keys: 'ok', 'bbox', 'color', 'name'
        
    Returns:
        frame: Frame with drawn boxes
    """
    
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
        return frame, None
    
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
    
    # Draw individual tracker boxes and collect points
    all_p1 = []
    all_p2 = []
    
    for t, weight in zip(successful, weights):
        bbox = t['bbox']
        color = t['color']
        name = t['name']
        
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        
        cv2.rectangle(frame, p1, p2, color, 1, 1)
        
        # Label each box with tracker name and weight
        label_pos = (p1[0], p1[1] - 5)
        cv2.putText(frame, f"{name} {int(weight*100)}%", label_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        all_p1.append(p1)
        all_p2.append(p2)
    
    # Calculate consensus center
    consensus_center = None
    if len(successful) > 0:
        # Weighted average of all corners
        mean_p1 = (
            int(sum(p[0] * w for p, w in zip(all_p1, weights))),
            int(sum(p[1] * w for p, w in zip(all_p1, weights)))
        )
        mean_p2 = (
            int(sum(p[0] * w for p, w in zip(all_p2, weights))),
            int(sum(p[1] * w for p, w in zip(all_p2, weights)))
        )
        
        # Draw consensus box in cyan, thicker
        cv2.rectangle(frame, mean_p1, mean_p2, (0, 255, 255), 1, 1)
        cv2.putText(frame, "Consensus", (mean_p1[0], mean_p1[1] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Calculate center of consensus box
        consensus_center = (
            (mean_p1[0] + mean_p2[0]) // 2,
            (mean_p1[1] + mean_p2[1]) // 2
        )
    
    return frame, consensus_center

'''
/////////////////////////////////////////////
#2 INTERPOLATION -> DRAW THE LINE
/////////////////////////////////////////////
'''
def interpolate_line_points(user_points, num_points):
    """
    Interpolate user-drawn line to get evenly spaced points.
    """
    user_points = np.array(user_points)
    
    if len(user_points) < 4:
        # Not enough points for spline, use linear interpolation
        total_length = 0
        for i in range(len(user_points) - 1):
            total_length += np.linalg.norm(user_points[i+1] - user_points[i])
        
        step = total_length / (num_points - 1)
        interpolated = [user_points[0]]
        current_length = 0
        
        for i in range(len(user_points) - 1):
            p1 = user_points[i]
            p2 = user_points[i + 1]
            segment_length = np.linalg.norm(p2 - p1)
            
            while current_length + segment_length >= step * len(interpolated) and len(interpolated) < num_points:
                t = (step * len(interpolated) - current_length) / segment_length
                new_point = p1 + t * (p2 - p1)
                interpolated.append(new_point)
            
            current_length += segment_length
        
        if len(interpolated) < num_points:
            interpolated.append(user_points[-1])
        
        return np.array(interpolated[:num_points])
    
    # Use spline interpolation for smooth curve
    x = user_points[:, 0]
    y = user_points[:, 1]
    
    tck, u = splprep([x, y], s=0, k=min(3, len(user_points)-1))
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)
    
    return np.column_stack([x_new, y_new])


'''
/////////////////////////////////////////////
#3 REMOVE OUTLIERS WHEN DRAWING THE LINE
/////////////////////////////////////////////
'''
def remove_outliers_ransac(centers, threshold=30):
    """
    Remove outliers using RANSAC curve fitting.
    
    Args:
        centers: List of (x, y) points
        threshold: Distance threshold for outliers (pixels)
    
    Returns:
        Filtered list of centers with outliers removed
    """
    if len(centers) < 4:
        return centers
    
    centers_array = np.array(centers)
    
    # Sort by x-coordinate for polynomial fitting
    sorted_indices = np.argsort(centers_array[:, 0])
    sorted_centers = centers_array[sorted_indices]
    
    x = sorted_centers[:, 0]
    y = sorted_centers[:, 1]
    
    # Fit polynomial (degree 2 or 3 depending on number of points)
    degree = min(3, len(centers) - 1)
    
    try:
        # Use RANSAC-like approach: fit multiple times and find consensus
        best_inliers = []
        best_inlier_count = 0
        
        for _ in range(10):  # Try 10 random subsets
            # Randomly sample subset
            if len(centers) > degree + 1:
                sample_idx = np.random.choice(len(centers), degree + 1, replace=False)
            else:
                sample_idx = range(len(centers))
            
            # Fit polynomial on sample
            coeffs = np.polyfit(x[sample_idx], y[sample_idx], degree)
            poly = np.poly1d(coeffs)
            
            # Calculate distances to fitted curve
            predicted_y = poly(x)
            distances = np.abs(y - predicted_y)
            
            # Find inliers
            inliers = distances < threshold
            inlier_count = np.sum(inliers)
            
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers
        
        # Return points that are inliers
        filtered_centers = sorted_centers[best_inliers]
        
        # Restore original order
        original_order_centers = []
        for idx in sorted_indices:
            if best_inliers[np.where(sorted_indices == idx)[0][0]]:
                original_order_centers.append(centers[idx])
        
        return original_order_centers
    
    except:
        return centers



'''
/////////////////////////////////////////////
#4 DRAW THE SMOOTH LINE AFTER INTERPOLATION
/////////////////////////////////////////////
'''

def draw_smooth_line(frame, centers, color=(0, 255, 0), thickness=2):
    """
    Draw a smooth curve that passes EXACTLY through all center points.
    Uses Catmull-Rom spline interpolation.
    """
    if len(centers) < 2:
        return
    
    centers_array = np.array(centers, dtype=np.float32)
    
    if len(centers) == 2:
        # Just draw a straight line
        cv2.line(frame, tuple(centers[0]), tuple(centers[1]), color, thickness, cv2.LINE_AA)
        return
    
    if len(centers) == 3:
        # For 3 points, use Catmull-Rom with duplicated endpoints
        points = [centers[0], centers[0], centers[1], centers[2], centers[2]]
    else:
        # For 4+ points, duplicate first and last for proper end behavior
        points = [centers[0]] + centers + [centers[-1]]
    
    curve_points = []
    
    # Catmull-Rom spline - guarantees passing through all control points
    for i in range(1, len(points) - 2):
        p0 = np.array(points[i-1], dtype=np.float32)
        p1 = np.array(points[i], dtype=np.float32)
        p2 = np.array(points[i+1], dtype=np.float32)
        p3 = np.array(points[i+2], dtype=np.float32)
        
        # Generate points along this segment
        num_points = 30  # More = smoother
        for t in np.linspace(0, 1, num_points, endpoint=(i == len(points) - 3)):
            t2 = t * t
            t3 = t2 * t
            
            # Catmull-Rom formula (guaranteed to pass through p1 and p2)
            point = 0.5 * (
                (2 * p1) +
                (-p0 + p2) * t +
                (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
                (-p0 + 3*p1 - 3*p2 + p3) * t3
            )
            curve_points.append(point.astype(np.int32))
    
    if curve_points:
        curve_points = np.array(curve_points)
        cv2.polylines(frame, [curve_points], False, color, thickness, cv2.LINE_AA)
    else:
        # Fallback
        points_array = np.array(centers, dtype=np.int32)
        cv2.polylines(frame, [points_array], False, color, thickness, cv2.LINE_AA)


if __name__ == '__main__':
    # Read video
    video = cv2.VideoCapture("videos/Lapchole/Lapchole3.mp4")
    
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    
    # Read first frame
    ok, frame = video.read()
    frame = cv2.resize(frame, None, fx=FRAME_RESIZE, fy=FRAME_RESIZE, interpolation=cv2.INTER_LINEAR)

    if not ok:
        print('Cannot read video file')
        sys.exit()
    
    # Interactive line drawing
    print("Instructions:")
    print("- Click and drag to draw a line along the structure")
    print("- Press ENTER when done")
    print("- Press 'C' to clear and restart")
    
    state = {
        'drawing': False,
        'user_points': [],
        'display_frame': frame.copy()
    }
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state['drawing'] = True
            state['user_points'].append([x, y])
        elif event == cv2.EVENT_MOUSEMOVE and state['drawing']:
            if len(state['user_points']) == 0 or np.linalg.norm(np.array([x, y]) - np.array(state['user_points'][-1])) > 3:
                state['user_points'].append([x, y])
                state['display_frame'] = frame.copy()
                if len(state['user_points']) > 1:
                    pts = np.array(state['user_points'], dtype=np.int32)
                    cv2.polylines(state['display_frame'], [pts], False, (0, 255, 0), 2)
                for pt in state['user_points']:
                    cv2.circle(state['display_frame'], tuple(pt), 2, (0, 0, 255), -1)
                cv2.imshow('Draw Line', state['display_frame'])
        elif event == cv2.EVENT_LBUTTONUP:
            state['drawing'] = False
    
    cv2.namedWindow('Draw Line')
    cv2.setMouseCallback('Draw Line', mouse_callback)
    cv2.imshow('Draw Line', state['display_frame'])
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ENTER
            break
        elif key == ord('c') or key == ord('C'):
            state['user_points'] = []
            state['display_frame'] = frame.copy()
            cv2.imshow('Draw Line', state['display_frame'])
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            sys.exit()
    
    cv2.destroyAllWindows()
    
    if len(state['user_points']) < 2:
        print("Error: Need at least 2 points")
        sys.exit()
    
    cv2.destroyAllWindows()
    
    if len(state['user_points']) < 2:
        print("Error: Need at least 2 points")
        sys.exit()
    
    # Configuration

    
    # Interpolate points along the line
    box_centers = interpolate_line_points(state['user_points'], NUM_POINTS)
    print(f"Created {len(box_centers)} tracking boxes")
    
    # Initialize trackers for each box
    box_trackers = []
    
    for center in box_centers:
        x, y = center
        bbox = (
            int(x - BOX_SIZE // 2),
            int(y - BOX_SIZE // 2),
            BOX_SIZE,
            BOX_SIZE
        )
        
        # Create ensemble for this box
        tracker1 = cv2.legacy.TrackerKCF_create()
        tracker2 = cv2.legacy.TrackerCSRT_create()
        tracker3 = cv2.legacy.TrackerMedianFlow_create()
        
        tracker1.init(frame, bbox)
        tracker2.init(frame, bbox)
        tracker3.init(frame, bbox)
        
        box_trackers.append({
            'trackers': [
                {'tracker': tracker1, 'name': 'KCF'},
                {'tracker': tracker2, 'name': 'CSRT'},
                {'tracker': tracker3, 'name': 'MEDIANFLOW'}
            ],
            'last_center': (int(x), int(y))
        })
    
    # Reset video to second frame
    video.set(cv2.CAP_PROP_POS_FRAMES, 1)
    
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
        
        # Track each box
        new_centers = []
        
        for box_ensemble in box_trackers:
            # Update each tracker in the ensemble
            ok1, bbox1 = box_ensemble['trackers'][0]['tracker'].update(frame)
            ok2, bbox2 = box_ensemble['trackers'][1]['tracker'].update(frame)
            ok3, bbox3 = box_ensemble['trackers'][2]['tracker'].update(frame)
            
            trackers_data = [
                {'ok': ok1, 'bbox': bbox1, 'color': (255, 0, 0), 'name': 'KCF'},
                {'ok': ok2, 'bbox': bbox2, 'color': (0, 255, 0), 'name': 'CSRT'},
                {'ok': ok3, 'bbox': bbox3, 'color': (0, 0, 255), 'name': 'MEDIANFLOW'}
            ]
            
            # Get consensus for this box (don't draw on frame yet)
            temp_frame = frame.copy()
            _, consensus_center = draw_tracked_boxes(temp_frame, trackers_data)
            
            if consensus_center is not None:
                new_centers.append(consensus_center)
                box_ensemble['last_center'] = consensus_center
            else:
                new_centers.append(box_ensemble['last_center'])

            
        
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        filtered_centers = remove_outliers_ransac(new_centers)

        
        # Draw smooth line through all centers
        draw_smooth_line(frame, filtered_centers, color=(0, 255, 0), thickness=3)
        
        # Draw boxes at each center
        for center in new_centers:
            x, y = center
            top_left = (x - BOX_SIZE // 2, y - BOX_SIZE // 2)
            bottom_right = (x + BOX_SIZE // 2, y + BOX_SIZE // 2)
            cv2.circle(frame, (x, y), bottom_right, (255, 0, 0), 1)
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        
        cv2.imshow("Tracking", frame)
        
        k = cv2.waitKey(22) & 0xff
        if k == 27:  # ESC
            break
    
    video.release()
    cv2.destroyAllWindows()