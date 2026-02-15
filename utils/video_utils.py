import cv2

def read_video(video_path):
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")
    frames=[]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def _get_writer_and_codec(output_video_path, frame_size, fps):
    if output_video_path.lower().endswith(".mp4"):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
        if writer.isOpened():
            return writer

    # Fallback for .avi or unsupported mp4 backend.
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    return writer

def save_video(output_video_frames,output_video_path):
    if not output_video_frames:
        raise ValueError("No frames to write. output_video_frames is empty.")
    frame_size = (output_video_frames[0].shape[1], output_video_frames[0].shape[0])
    out = _get_writer_and_codec(output_video_path, frame_size, 24)
    if not out.isOpened():
        raise RuntimeError(f"Unable to create output video: {output_video_path}")
    for frame in output_video_frames:
        out.write(frame)
    out.release()    


