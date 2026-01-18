from multibox_kcf_csrt import active_line_tracking, multi_line_select, init_video, start_line_tracking, next_frame, draw_lines

import cv2
import sys


if __name__ == '__main__':
    # Read video
    frame, video = init_video("videos/Echo/echo1.mp4")
    
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    
    # Read first frame
    ok, frame = video.read()
    frame = cv2.resize(frame, None, fx=8, fy=8, interpolation=cv2.INTER_LINEAR)

    all_bboxes = multi_line_select(frame)

    trackers, flags = start_line_tracking(all_bboxes, frame)

    if not ok:
        print('Cannot read video file')
        sys.exit()
    
    while True:
        ok, frame = next_frame(frame, video)

        if not ok: break

        
        timer = cv2.getTickCount()


        # count = 0
        # for line in all_bboxes:
        #     drawn_centres, count = active_line_tracking(trackers, frame, line, count, flags=flags)
        #     draw_lines(frame, drawn_centres)
        # count = 0
        # for line in all_bboxes:
        #     drawn_centres, count = active_line_tracking(trackers, frame, line, count, flags=flags)
        #     draw_lines(frame, drawn_centres)
        
        drawn_centres, count = active_line_tracking(trackers, frame)
        print(drawn_centres)

        draw_lines(frame, drawn_centres)
        cv2.imshow("Tracking", frame)


        k = cv2.waitKey(22) & 0xff
        if k == 27:  # ESC
            break

    video.release()
    cv2.destroyAllWindows()
