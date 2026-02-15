import argparse
from pathlib import Path

from utils.video_utils import read_video, save_video
from Trackers import PlayerTracker, BallTracker
from Court_Line_Detector import CourtLineDetector
from Mini_court import MiniCourt
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description="Tennis analytics video pipeline.")
    parser.add_argument("--input-video", default="Input_Videos/input_video.mp4")
    parser.add_argument("--output-video", default="Output_Videos/output_video.avi")
    parser.add_argument("--player-model", default="yolov8x.pt")
    parser.add_argument("--ball-model", default="Models/last.pt")
    parser.add_argument("--court-model", default="Models/keypoints_model.pth")
    parser.add_argument("--player-stub", default="Tracker_Stubs/player_detections.pkl")
    parser.add_argument("--ball-stub", default="Tracker_Stubs/ball_detections.pkl")
    parser.add_argument(
        "--use-stubs",
        action="store_true",
        help="Load player/ball detections from pickle files instead of running detection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_video_path = Path(args.input_video)
    if not input_video_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    video_frames = read_video(str(input_video_path))
    if not video_frames:
        raise RuntimeError(f"No frames were read from: {input_video_path}")

    player_tracker = PlayerTracker(model_path=args.player_model)
    ball_tracker = BallTracker(model_path=args.ball_model)

    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=args.use_stubs,
        stub_path=args.player_stub,
    )
    ball_detections = ball_tracker.detect_frames(
        video_frames,
        read_from_stub=args.use_stubs,
        stub_path=args.ball_stub,
    )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    court_line_detector = CourtLineDetector(args.court_model)
    court_keypoints = court_line_detector.predict(video_frames[0])
    player_detections = player_tracker.choose_and_filter_players(
        court_keypoints, player_detections
    )

    mini_court = MiniCourt(video_frames[0])

    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_video(
        output_video_frames, court_keypoints
    )
    output_video_frames = mini_court.draw_mini_court(
        output_video_frames,
        player_detections=player_detections,
        ball_detections=ball_detections,
        court_keypoints=court_keypoints,
    )

    for i, frame in enumerate(output_video_frames):
        cv2.putText(
            frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

    output_video_path = Path(args.output_video)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(output_video_frames, str(output_video_path))


if __name__ == "__main__":
    main()
