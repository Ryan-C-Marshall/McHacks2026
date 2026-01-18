from lines_kcf_csrt import fast_track, multi_line_select, init_video, start_line_tracking, next_frame, draw_lines

import cv2
import sys


if __name__ == '__main__':
    # Read video
    frame, video = init_video("videos/Echo/echo1.mp4")
    

    

    all_bboxes = multi_line_select(frame)
    print(all_bboxes)

    trackers, flags = start_line_tracking(all_bboxes, frame)


    while True:
        ok, frame = next_frame(frame, video)

        if not ok: break

        
        timer = cv2.getTickCount()
        
        drawn_centres, count = fast_track(trackers, frame)
        print(drawn_centres)

        draw_lines(frame, drawn_centres)
        cv2.imshow("Tracking", frame)




        k = cv2.waitKey(22) & 0xff
        if k == 27:  # ESC
            break

    video.release()
    cv2.destroyAllWindows()
