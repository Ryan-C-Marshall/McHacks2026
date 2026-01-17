import cv2
import sys

FRAME_RESIZE = 8

if __name__ == '__main__':
    # Create trackers using legacy module
    tracker1 = cv2.legacy.TrackerKCF_create()
    tracker2 = cv2.legacy.TrackerCSRT_create()
    tracker3 = cv2.legacy.TrackerMedianFlow_create()


    
    tracker_type1 = "KCF"  # Add these variables
    tracker_type2 = "CSRT"
    
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
    
    
    while True:
        ok, frame = video.read()
        frame = cv2.resize(frame, None, fx=FRAME_RESIZE, fy=FRAME_RESIZE, interpolation=cv2.INTER_LINEAR)

        if not ok:
            break
        
        timer = cv2.getTickCount()
        
        # Update trackers
        ok1, bbox1 = tracker1.update(frame)
        ok2, bbox2 = tracker2.update(frame)
        ok3, bbox3 = tracker3.update(frame)

        
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        
        if ok1 and ok2 and ok3:
            # Draw individual tracker boxes
            p1_1 = (int(bbox1[0]), int(bbox1[1]))
            p1_2 = (int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3]))
            p2_1 = (int(bbox2[0]), int(bbox2[1]))
            p2_2 = (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))
            p3_1 = (int(bbox3[0]), int(bbox3[1]))
            p3_2 = (int(bbox3[0] + bbox3[2]), int(bbox3[1] + bbox3[3]))
            
            cv2.rectangle(frame, p1_1, p1_2, (255, 0, 0), 2, 1)
            cv2.rectangle(frame, p2_1, p2_2, (0, 255, 0), 2, 1)
            cv2.rectangle(frame, p3_1, p3_2, (0, 0, 255), 2, 1)
            
            # Fusion - average the bounding boxes
            mean_p1 = (int((p1_1[0] + p2_1[0] + p3_1[0]) / 3), int((p1_1[1] + p2_1[1]+ p3_1[1]) / 3))
            mean_p2 = (int((p1_2[0] + p2_2[0]+ p3_2[0]) / 3), int((p1_2[1] + p2_2[1]+ p3_1[1]) / 3))
            cv2.rectangle(frame, mean_p1, mean_p2, (0, 255, 255), 3, 1)  # Thicker for consensus

        
        
        elif ok1 and ok2:
            # Draw individual tracker boxes
            p1_1 = (int(bbox1[0]), int(bbox1[1]))
            p1_2 = (int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3]))
            p2_1 = (int(bbox2[0]), int(bbox2[1]))
            p2_2 = (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))
            
            
            cv2.rectangle(frame, p1_1, p1_2, (255, 0, 0), 2, 1)
            cv2.rectangle(frame, p2_1, p2_2, (0, 255, 0), 2, 1)
            
            # Fusion - average the bounding boxes
            mean_p1 = (int((p1_1[0] + p2_1[0]) / 2), int((p1_1[1] + p2_1[1]) / 2))
            mean_p2 = (int((p1_2[0] + p2_2[0]) / 2), int((p1_2[1] + p2_2[1]) / 2))
            cv2.rectangle(frame, mean_p1, mean_p2, (0, 255, 255), 3, 1)  # Thicker for consensus

        elif ok1 and ok3:
            # Draw individual tracker boxes
            p1_1 = (int(bbox1[0]), int(bbox1[1]))
            p1_2 = (int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3]))
            p3_1 = (int(bbox3[0]), int(bbox3[1]))
            p3_2 = (int(bbox3[0] + bbox3[2]), int(bbox3[1] + bbox3[3]))
            
            
            cv2.rectangle(frame, p1_1, p1_2, (255, 0, 0), 2, 1)
            cv2.rectangle(frame, p3_1, p3_2, (0, 0, 255), 2, 1)
            
            # Fusion - average the bounding boxes
            mean_p1 = (int((p1_1[0] + p3_1[0]) / 2), int((p1_1[1] + p3_1[1]) / 2))
            mean_p2 = (int((p1_2[0] + p3_2[0]) / 2), int((p1_2[1] + p3_2[1]) / 2))
            cv2.rectangle(frame, mean_p1, mean_p2, (0, 255, 255), 3, 1)  # Thicker for consensus
        
        elif ok2 and ok3:
            # Draw individual tracker boxes
            p2_1 = (int(bbox2[0]), int(bbox2[1]))
            p2_2 = (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))
            p3_1 = (int(bbox3[0]), int(bbox3[1]))
            p3_2 = (int(bbox3[0] + bbox3[2]), int(bbox3[1] + bbox3[3]))
            
            
            cv2.rectangle(frame, p2_1, p2_2, (255, 0, 0), 2, 1)
            cv2.rectangle(frame, p3_1, p3_2, (0, 255, 0), 2, 1)
            
            # Fusion - average the bounding boxes
            mean_p1 = (int((p2_1[0] + p3_1[0]) / 2), int((p2_1[1] + p3_1[1]) / 2))
            mean_p2 = (int((p2_2[0] + p3_2[0]) / 2), int((p2_2[1] + p3_2[1]) / 2))
            cv2.rectangle(frame, mean_p1, mean_p2, (0, 255, 255), 3, 1)  # Thicker for consensus
        
        elif ok1:
            p1_1 = (int(bbox1[0]), int(bbox1[1]))
            p1_2 = (int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3]))
            
            cv2.rectangle(frame, p1_1, p1_2, (255, 0, 0), 2, 1)
        elif ok2:
            p2_1 = (int(bbox2[0]), int(bbox2[1]))
            p2_2 = (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))
            
            cv2.rectangle(frame, p2_1, p2_2, (0, 255, 0), 2, 1)
        
        
        elif ok3:
            p3_1 = (int(bbox3[0]), int(bbox3[1]))
            p3_2 = (int(bbox3[0] + bbox3[2]), int(bbox3[1] + bbox3[3]))
            
            cv2.rectangle(frame, p3_1, p3_2, (0, 0, 255), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure detected", (50, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        
        
        # Display info
        
        cv2.putText(frame, f"{tracker_type1} + {tracker_type2}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 170, 50), 2)
        cv2.putText(frame, f"FPS : {int(fps)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 170, 50), 2)
        
        cv2.imshow("Tracking", frame)
        
        k = cv2.waitKey(22) & 0xff
        if k == 27:  # ESC
            break
        

    video.release()
    cv2.destroyAllWindows()