import cv2
import sys
import math
from polars import col

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

def contains_circle(bbox_1, bbox_2, allowance=ALLOWANCE):
    """
    Check if circle_2's center is within circle_1's radius.
    Converts bounding boxes to circles automatically.
    
    Args:
        bbox_1: tuple (x, y, width, height)
        bbox_2: tuple (x, y, width, height)
        allowance: float, percentage allowance (default 0.1 = 10%)
    
    Returns:
        bool: True if circle_2's center is inside circle_1
    """
    # Convert bbox to circle centers
    x1, y1, w1, h1 = bbox_1
    x2, y2, w2, h2 = bbox_2
    
    # Calculate centers
    c1_x = x1 + w1 / 2
    c1_y = y1 + h1 / 2
    c2_x = x2 + w2 / 2
    c2_y = y2 + h2 / 2
    
    # Use average of width and height as diameter, then divide by 2 for radius
    r1 = (w1 + h1) / 4
    
    # Expand circle_1's radius by allowance percentage
    effective_radius = r1 * (allowance)
    
    # Calculate distance between centers
    distance = math.sqrt((c2_x - c1_x)**2 + (c2_y - c1_y)**2)
    
    # Check if center of circle_2 is within effective radius of circle_1
    return distance <= effective_radius





def not_wider(box_1, box_2, size_allowance=SIZE_ALLOWANCE):
    # Check if box_2 is wider than box_1 by a factor of width_allowance
    x1_1, y1_1, w1, h1 = box_1
    x2_1, y2_1, w2, h2 = box_2

    width_allowance = w1 * size_allowance  # Convert allowance to pixels based on width of box_1
    height_allowance = h1 * size_allowance  # Convert allowance to pixels based on height of box_1

    if w2 > width_allowance or h2 > height_allowance:
        return False
    return True

def test_tester():
    """Test cases for the tester function"""
    # Test 1: Center of box_2 clearly inside box_1
    assert contains((0, 0, 100, 100), (25, 25, 20, 20)) == True, "Test 1 failed: Center inside"
    
    # Test 2: Center of box_2 clearly outside box_1
    assert contains((0, 0, 100, 100), (200, 200, 20, 20)) == False, "Test 2 failed: Center outside"
    
    # Test 3: Center of box_2 on edge of box_1 (should be True with <= operator)
    assert contains((0, 0, 100, 100), (100, 100, 20, 20)) == True, "Test 3 failed: Center on edge"
    
    # Test 4: Boxes completely separate
    assert contains((0, 0, 50, 50), (100, 100, 30, 30)) == False, "Test 4 failed: Completely separate"
    
    # Test 5: box_2 partially overlaps but center is inside
    assert contains((0, 0, 100, 100), (80, 80, 40, 40)) == True, "Test 5 failed: Partial overlap with center inside"

    # Test 6: box_2 much larger but center inside box_1
    assert contains((20, 20, 60, 60), (10, 10, 100, 100)) == True, "Test 6 failed: Large box with center inside"

    print("All tests passed!")


if __name__ == '__main__':
    # Create trackers using legacy module
    tracker1 = cv2.TrackerKCF_create()
    tracker2 = cv2.legacy.TrackerCSRT_create()
    tracker3 = cv2.legacy.TrackerMedianFlow_create()

    tracker_type1 = "KCF"  # Add these variables
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
    cv2.putText(frame, f"{message}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.namedWindow('ROI Selector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ROI Selector', 1280, 720)

    bbox = (596, 535, 157, 114)
    p1_1 = (int(bbox[0]), int(bbox[1]))
    p1_2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1_1, p1_2, (255, 0, 0), 2, 1)
    cv2.imshow('ROI Selector', frame)

    key = cv2.waitKey(0) & 0xff
    x = (key == ord('y'))  # x is True if 'y' is pressed, False otherwise
    



    if x:
        bbox = cv2.selectROI('ROI Selector', frame, False)
    print(bbox)
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
        ok1, dummy1 = tracker1.update(frame)
        if ok1:
            bbox1 = dummy1
        ok2, dummy2 = tracker2.update(frame)
        if ok2:
            bbox2 = dummy2
        ok3, dummy3 = tracker3.update(frame)
        if ok3:
            bbox3 = dummy3

        
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        
        
        if ok1:
            # Draw bounding box from tracker 1
            p1_1 = (int(bbox1[0]), int(bbox1[1]))
            p1_2 = (int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3]))
            cv2.rectangle(frame, p1_1, p1_2, (255, 0, 0), 2, 1)
    


            cv2.putText(frame, f"{tracker_type1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        elif ok2 and contains(bbox1, bbox2) and not_wider(bbox1, bbox2):
            print(f"bbox1 = {bbox1}")
            # Draw bounding box from tracker 2
            p2_1 = (int(bbox2[0]), int(bbox2[1]))
            p2_2 = (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))
            cv2.rectangle(frame, p2_1, p2_2, (0, 255, 0), 2, 1)


            cv2.putText(frame, f"{tracker_type2}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)




        elif ok3 and contains(bbox2, bbox3) and not_wider(bbox2, bbox3):
            # Draw bounding box from tracker 3
            p3_1 = (int(bbox3[0]), int(bbox3[1]))
            p3_2 = (int(bbox3[0] + bbox3[2]), int(bbox3[1] + bbox3[3]))
            cv2.rectangle(frame, p3_1, p3_2, (0, 0, 255), 2, 1)
            cv2.putText(frame, f"{tracker_type3}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        
        else:
            cv2.putText(frame, "Tracking failure detected", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            p1_1 = (int(bbox1[0]), int(bbox1[1]))
            p1_2 = (int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3]))
            cv2.rectangle(frame, p1_1, p1_2, (255, 0, 0), 2, 1)

            p2_1 = (int(bbox2[0]), int(bbox2[1]))
            p2_2 = (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))
            cv2.rectangle(frame, p2_1, p2_2, (0, 255, 0), 2, 1)


            p3_1 = (int(bbox3[0]), int(bbox3[1]))
            p3_2 = (int(bbox3[0] + bbox3[2]), int(bbox3[1] + bbox3[3]))
            cv2.rectangle(frame, p3_1, p3_2, (0, 0, 255), 2, 1)



        
        
        # Display info
        
        # cv2.putText(frame, f"{tracker_type1} + {tracker_type2}", (10, 30), 
        # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 170, 50), 2)
        cv2.putText(frame, f"FPS : {int(fps)}", (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (204, 0, 204), 2)

        cv2.putText(frame, f"Box12, Box23, width12, width23: {contains(bbox1, bbox2), contains(bbox2, bbox3), not_wider(bbox1, bbox2), not_wider(bbox2, bbox3)}", (10, 90), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (204, 0, 204), 2)
        
        cv2.imshow("Tracking", frame)
        
        k = cv2.waitKey(22) & 0xff
        if k == 27:  # ESC
            break
        

    video.release()
    cv2.destroyAllWindows()