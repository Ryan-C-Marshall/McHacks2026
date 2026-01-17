import cv2
import sys

def draw_tracked_boxes(frame, trackers_data):
    
    # Filter successful trackers
    successful = [(bbox, color) for ok, bbox, color in trackers_data if ok]
    
    if not successful:
        # No tracking success
        cv2.putText(frame, "Tracking failure detected", (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        return frame
    
    all_p1 = []
    all_p2 = []
    
    # Draw individual tracker boxes
    
    for bbox, color in successful:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        
        cv2.rectangle(frame, p1, p2, color, 2, 1)
        
        all_p1.append(p1)
        all_p2.append(p2)
    
    if len(successful) > 0:
        # Average all corners
        mean_p1 = (
            int(sum(p[0] for p in all_p1) / len(all_p1)),
            int(sum(p[1] for p in all_p1) / len(all_p1))
        )
        mean_p2 = (
            int(sum(p[0] for p in all_p2) / len(all_p2)),
            int(sum(p[1] for p in all_p2) / len(all_p2))
        )
        
        # Draw consensus box in cyan, thicker
        cv2.rectangle(frame, mean_p1, mean_p2, (0, 255, 255), 3, 1)
    
    return frame

if __name__ == '__main__':
    # Create trackers using legacy module
    tracker1 = cv2.legacy.TrackerKCF_create()
    tracker2 = cv2.legacy.TrackerCSRT_create()
    tracker3 = cv2.legacy.TrackerMedianFlow_create()


    
    tracker_type1 = "KCF"  # Add these variables
    tracker_type2 = "CSRT"
    
    # Read video
    video = cv2.VideoCapture("echo1.mp4")
    
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    
    # Read first frame
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    
    # Select ROI
    cv2.namedWindow('ROI Selector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ROI Selector', 1280, 720)
    bbox = cv2.selectROI('ROI Selector', frame, False)
    cv2.destroyWindow('ROI Selector')
    
    # Initialize both trackers with same bbox
    ok1 = tracker1.init(frame, bbox)
    ok2 = tracker2.init(frame, bbox)  # Fix: was tracker1 again
    ok3 = tracker3.init(frame, bbox)
    
    # Create tracking window
    cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Tracking', 1280, 720)

    while True:
        ok, frame = video.read()
        if not ok:
            break
        
        timer = cv2.getTickCount()
        
        # Update all trackers
        ok1, bbox1 = tracker1.update(frame)
        ok2, bbox2 = tracker2.update(frame)
        ok3, bbox3 = tracker3.update(frame)
        
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        
        # Define tracker data: (success, bbox, color)
        trackers_data = [
            (ok1, bbox1, (255, 0, 0)),    # Blue for tracker 1
            (ok2, bbox2, (0, 255, 0)),    # Green for tracker 2
            (ok3, bbox3, (0, 0, 255))     # Red for tracker 3
        ]
        
        # Draw all boxes and consensus
        frame = draw_tracked_boxes(frame, trackers_data)
    
        
        cv2.imshow("Tracking", frame)
        
        k = cv2.waitKey(22) & 0xff
        if k == 27:  # ESC
            break
    
        

    video.release()
    cv2.destroyAllWindows()





