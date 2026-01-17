import cv2
import sys

FRAME_RESIZE = 8

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
        # KCF gets 70%, distribute remaining 30%
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
    
    # Draw consensus box ONLY if we have multiple trackers
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
    
    return frame



if __name__ == '__main__':
    # Create trackers using legacy module
    tracker1 = cv2.legacy.TrackerKCF_create()
    tracker2 = cv2.legacy.TrackerCSRT_create()
    tracker3 = cv2.legacy.TrackerMedianFlow_create()


    
    tracker_type1 = "KCF"  # Add these variables
    tracker_type2 = "CSRT"
    tracker_type2 = "MedianFlow"
    
    
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
    
    # Select ROI
    cv2.namedWindow('ROI Selector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ROI Selector', 1280, 720)
    bbox = (611, 534, 156, 120)
    # bbox = cv2.selectROI('ROI Selector', frame, False)
    cv2.destroyWindow('ROI Selector')
    
    # Initialize both trackers with same bbox
    ok1 = tracker1.init(frame, bbox)
    ok2 = tracker2.init(frame, bbox)  # Fix: was tracker1 again
    ok3 = tracker3.init(frame, bbox)
    
    # Create tracking window
    cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Tracking', 1280, 720)

    # Usage in your main loop:
    while True:
        ok, frame = video.read()
        frame = cv2.resize(frame, None, fx=FRAME_RESIZE, fy=FRAME_RESIZE, interpolation=cv2.INTER_LINEAR)

        if not ok:
            break
        
        timer = cv2.getTickCount()
        
        # Update all trackers
        ok1, bbox1 = tracker1.update(frame)  # KCF
        ok2, bbox2 = tracker2.update(frame)  # CSRT
        ok3, bbox3 = tracker3.update(frame)  # MEDIANFLOW
        
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        
        # Define tracker data as list of dictionaries
        trackers_data = [
            {'ok': ok1, 'bbox': bbox1, 'color': (255, 0, 0), 'name': 'KCF'},
            {'ok': ok2, 'bbox': bbox2, 'color': (0, 255, 0), 'name': 'CSRT'},
            {'ok': ok3, 'bbox': bbox3, 'color': (0, 0, 255), 'name': 'MEDIANFLOW'}
        ]
        
        # Draw all boxes and weighted consensus
        frame = draw_tracked_boxes(frame, trackers_data)
        
        # Display info
        active_count = sum([ok1, ok2, ok3])
        '''
        cv2.putText(frame, f"Trackers: {active_count}/3 active", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 170, 50), 2)
        cv2.putText(frame, f"FPS : {int(fps)}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 170, 50), 2)
        '''
        cv2.imshow("Tracking", frame)
        
        
        k = cv2.waitKey(22) & 0xff
        if k == 27:  # ESC
            break
    
        

    video.release()
    cv2.destroyAllWindows()





