from kcf_to_csrt import contains, not_wider, FRAME_RESIZE, ALLOWANCE, SIZE_ALLOWANCE
import cv2
import sys

tracker_type1 = "KCF"
tracker_type2 = "CSRT"
tracker_type3 = "MEDIANFLOW"
tracker_types = [tracker_type1, tracker_type2, tracker_type3]

def display(bbox1, bbox2, bbox3, ok1, ok2, ok3, frame, fps):
        if ok1:
            # Draw bounding box from tracker 1
            p1_1 = (int(bbox1[0]), int(bbox1[1]))
            p1_2 = (int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3]))
            cv2.rectangle(frame, p1_1, p1_2, (255, 0, 0), 2, 1)
    


            cv2.putText(frame, f"{tracker_types[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        elif ok2 and contains(bbox1, bbox2) and not_wider(bbox1, bbox2):
            print(f"bbox1 = {bbox1}")
            # Draw bounding box from tracker 2
            p2_1 = (int(bbox2[0]), int(bbox2[1]))
            p2_2 = (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))
            cv2.rectangle(frame, p2_1, p2_2, (0, 255, 0), 2, 1)


            cv2.putText(frame, f"{tracker_types[1]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)




        elif ok3 and contains(bbox2, bbox3) and not_wider(bbox2, bbox3):
            # Draw bounding box from tracker 3
            p3_1 = (int(bbox3[0]), int(bbox3[1]))
            p3_2 = (int(bbox3[0] + bbox3[2]), int(bbox3[1] + bbox3[3]))
            cv2.rectangle(frame, p3_1, p3_2, (0, 0, 255), 2, 1)
            cv2.putText(frame, f"{tracker_types[2]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        
        else:
            cv2.putText(frame, "Tracking failure detected", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)




        
        
        # Display info
        
        # cv2.putText(frame, f"{tracker_type1} + {tracker_type2}", (10, 30), 
        # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 170, 50), 2)
        cv2.putText(frame, f"FPS : {int(fps)}", (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (204, 0, 204), 2)

        cv2.putText(frame, f"Box12, Box23, width12, width23: {contains(bbox1, bbox2), contains(bbox2, bbox3), not_wider(bbox1, bbox2), not_wider(bbox2, bbox3)}", (10, 90), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (204, 0, 204), 2)
        
        cv2.imshow("Tracking", frame)
        

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

def initialize_trackers(bboxes, frame, trackers=[[], [],[]], ):
    for i, bbox in enumerate(bboxes):
        trackers[0].append(cv2.legacy.TrackerKCF_create())
        trackers[1].append(cv2.legacy.TrackerCSRT_create())
        trackers[2].append(cv2.legacy.TrackerMedianFlow_create())
        
        # Initialize trackers with bbox
        trackers[0][i].init(frame, bbox)
        trackers[1][i].init(frame, bbox)
        trackers[2][i].init(frame, bbox)

    return trackers
        

def gen_frame(path):
    video = cv2.VideoCapture(path)
    
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    
    # Read first frame
    ok, frame = video.read()
    frame = cv2.resize(frame, None, fx=FRAME_RESIZE, fy=FRAME_RESIZE, interpolation=cv2.INTER_LINEAR)

    return frame, video

def next_frame(frame, video):
    ok, frame = video.read()
    frame = cv2.resize(frame, None, fx=FRAME_RESIZE, fy=FRAME_RESIZE, interpolation=cv2.INTER_LINEAR)


    return ok, frame

if __name__ == '__main__':
        
    # Read video

    frame, video = gen_frame("videos/Echo/echo1.mp4")
    
    bboxes, frame = multi_select()
    
    num_boxes = len(bboxes)
    
    
    trackers = initialize_trackers(bboxes, frame)

    while True:
        ok, frame = next_frame(frame, video)
        
        if not ok: break
        
        timer = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        for i in range(num_boxes):
            # Update each tracker
            ok1, dummy1 = trackers[0][i].update(frame)
            if ok1:
                bbox1 = dummy1
            ok2, dummy2 = trackers[1][i].update(frame)
            if ok2:
                bbox2 = dummy2
            ok3, dummy3 = trackers[2][i].update(frame)
            if ok3:
                bbox3 = dummy3
            

            display(bbox1, bbox2, bbox3, ok1, ok2, ok3, frame, fps)

        k = cv2.waitKey(22) & 0xff
        if k == 27:  # ESC
            break

    video.release()
    cv2.destroyAllWindows()
        