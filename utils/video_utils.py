import cv2

def read_video(video_path): # function to read a video, loops over the video and adds each frame to a list.
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read() #reads the frame
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    print(f"Video saved to {output_video_path}")
