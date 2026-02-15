from ultralytics import YOLO
import cv2
import pickle
import numpy as np
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        if not player_detections:
            return []
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        if len(player_dict) <= 2:
            return list(player_dict.keys())

        # Prefer players that are on/near the detected court area, then pick one
        # nearest-side player and one far-side player.
        if len(court_keypoints) >= 8:
            court_polygon = np.array([
                (court_keypoints[0], court_keypoints[1]),
                (court_keypoints[2], court_keypoints[3]),
                (court_keypoints[6], court_keypoints[7]),
                (court_keypoints[4], court_keypoints[5]),
            ], dtype=np.float32)
            inside_or_near = []
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                anchor = (int((x1 + x2) / 2), int(y1 + 0.82 * (y2 - y1)))
                dist_to_poly = cv2.pointPolygonTest(court_polygon, anchor, True)
                if dist_to_poly >= -80:
                    inside_or_near.append((track_id, anchor[1]))

            if len(inside_or_near) >= 2:
                # Largest y => closer player, smallest y => far player.
                inside_or_near.sort(key=lambda x: x[1])
                far_player = inside_or_near[0][0]
                near_player = inside_or_near[-1][0]
                if far_player != near_player:
                    return [far_player, near_player]

        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)
            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i + 1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        distances.sort(key=lambda x: x[1])
        return [distances[0][0], distances[1][0]]

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            if box.id is None:
                continue
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = int(box.cls.tolist()[0])
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames
