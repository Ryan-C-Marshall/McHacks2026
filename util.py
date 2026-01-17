import cv2
import sys

def load_video_thumbnail(video_path, time_sec=0):
    """Load a thumbnail image from the video at the specified time (in seconds)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(fps * time_sec)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read frame.")
        return None

    return frame

def load_videos_from_directory(directory_path='videos/'):
    """Load all video file paths from the specified directory.
    
    Outputs them in the form:
    [{'title': 'Echo Ultrasound', 'desc': 'Track objects in echocardiography videos.', 'path': 'videos/Echo/echo1.mp4'}, ...]

    """
    import os
    video_list = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(root, file)
                title = os.path.splitext(file)[0].replace('_', ' ').title()
                desc = "Description for " + title  # Placeholder description
                video_list.append({'title': title, 'desc': desc, 'path': video_path})

    print("Loaded videos:", video_list)
    return video_list