# Tennis ML Project

End-to-end tennis video analysis pipeline with:
- Player tracking (`YOLOv8`)
- Ball detection + interpolation
- Court keypoint detection (`ResNet50`)
- Mini-court overlay rendering
- Annotated output video generation

## Project Structure
- `main.py`: Pipeline entrypoint
- `Trackers/`: Player and ball tracking modules
- `Court_Line_Detector/`: Court keypoint model inference
- `Mini_court/`: Court overlay drawing
- `utils/`: Video IO, bbox helpers, metric conversions
- `Models/`: Trained model artifacts
- `Input_Videos/`: Source videos
- `Output_Videos/`: Rendered outputs
- `Tracker_Stubs/`: Optional cached detections

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python main.py --input-video Input_Videos/input_video.mp4 --output-video Output_Videos/output_video.avi
```

Use cached detections:
```bash
python main.py --use-stubs --player-stub Tracker_Stubs/player_detections.pkl --ball-stub Tracker_Stubs/ball_detections.pkl
```

## Notes
- Ensure model files exist:
  - `yolov8x.pt`
  - `Models/last.pt`
  - `Models/keypoints_model.pth`
- For first runs, omit `--use-stubs` to generate fresh detections.
