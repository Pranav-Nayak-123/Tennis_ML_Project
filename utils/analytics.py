import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class HitEvent:
    frame: int
    time_sec: float
    hitter: str
    shot_type: str
    speed_kmh: float


class RallyAnalyzer:
    def __init__(self, fps=24):
        self.fps = fps

    @staticmethod
    def _bbox_center(bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _player_anchor(bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, y1 + 0.82 * (y2 - y1)

    @staticmethod
    def _distance(p1, p2):
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    @staticmethod
    def _project(mini_court, point, homography):
        projected = mini_court._project_point(point, homography)
        if projected is None:
            return None
        if not mini_court._is_inside_mini_court(projected, margin=100):
            return None
        return mini_court._clamp_to_mini_court(projected)

    @staticmethod
    def _assign_far_near(projected_players):
        if not projected_players:
            return None, None
        projected_players = sorted(projected_players, key=lambda p: p[1])
        if len(projected_players) == 1:
            return projected_players[0], None
        return projected_players[0], projected_players[-1]

    def _infer_shot_type(self, hitter, frame_idx, rally_start, speed_kmh, ball_x, near_player_x, far_player_x):
        if frame_idx - rally_start <= int(1.5 * self.fps) and hitter == "far" and speed_kmh >= 90:
            return "serve"

        if hitter == "near" and near_player_x is not None:
            return "forehand" if ball_x >= near_player_x else "backhand"
        if hitter == "far" and far_player_x is not None:
            return "forehand" if ball_x <= far_player_x else "backhand"
        return "unknown"

    def analyze(self, frames, player_detections, ball_detections, mini_court, court_keypoints):
        frame_count = len(frames)
        homography = mini_court._compute_homography(court_keypoints)
        if homography is None:
            raise RuntimeError("Unable to compute homography for analytics.")

        ball_mini_points = [None] * frame_count
        speed_kmh = [0.0] * frame_count
        near_player_x_img = [None] * frame_count
        far_player_x_img = [None] * frame_count
        near_player_mini = [None] * frame_count
        far_player_mini = [None] * frame_count

        meters_per_pixel = 1.0 / max(mini_court.pixels_per_meter, 1e-6)

        for i in range(frame_count):
            if i < len(player_detections):
                projected_players = []
                for _, bbox in player_detections[i].items():
                    projected = self._project(mini_court, self._player_anchor(bbox), homography)
                    if projected is not None:
                        projected_players.append(projected)

                far_pt, near_pt = self._assign_far_near(projected_players)
                far_player_mini[i] = far_pt
                near_player_mini[i] = near_pt

                # Map near/far image x from bbox anchors for shot-type side inference.
                if player_detections[i]:
                    anchors = [self._player_anchor(b) for b in player_detections[i].values()]
                    anchors = sorted(anchors, key=lambda p: p[1])
                    if anchors:
                        far_player_x_img[i] = anchors[0][0]
                    if len(anchors) > 1:
                        near_player_x_img[i] = anchors[-1][0]

            if i < len(ball_detections) and 1 in ball_detections[i]:
                ball_center = self._bbox_center(ball_detections[i][1])
                ball_mini_points[i] = self._project(mini_court, ball_center, homography)

            if i > 0 and ball_mini_points[i] is not None and ball_mini_points[i - 1] is not None:
                pixel_dist = self._distance(ball_mini_points[i], ball_mini_points[i - 1])
                mps = pixel_dist * meters_per_pixel * self.fps
                speed_kmh[i] = min(260.0, mps * 3.6)
            else:
                speed_kmh[i] = speed_kmh[i - 1] * 0.9 if i > 0 else 0.0

        # Hit detection from ball proximity to players + cooldown.
        hit_events = []
        hit_frames = []
        last_hit_frame = -10_000
        hit_cooldown = max(6, int(0.25 * self.fps))
        hit_dist_threshold = max(26, int(mini_court.court_drawing_width * 0.10))

        for i in range(frame_count):
            ball_pt = ball_mini_points[i]
            if ball_pt is None:
                continue

            far_pt = far_player_mini[i]
            near_pt = near_player_mini[i]
            if far_pt is None and near_pt is None:
                continue

            dist_far = self._distance(ball_pt, far_pt) if far_pt is not None else float("inf")
            dist_near = self._distance(ball_pt, near_pt) if near_pt is not None else float("inf")
            hitter = "far" if dist_far <= dist_near else "near"
            min_dist = min(dist_far, dist_near)

            if min_dist <= hit_dist_threshold and i - last_hit_frame >= hit_cooldown:
                hit_frames.append(i)
                last_hit_frame = i

        # Build rallies and infer shot types/outcomes.
        rallies = []
        rally_gap = int(2.0 * self.fps)
        if hit_frames:
            current_start = hit_frames[0]
            current_hits = [hit_frames[0]]
            for frame_idx in hit_frames[1:]:
                if frame_idx - current_hits[-1] > rally_gap:
                    rallies.append((current_start, current_hits[-1], current_hits))
                    current_start = frame_idx
                    current_hits = [frame_idx]
                else:
                    current_hits.append(frame_idx)
            rallies.append((current_start, current_hits[-1], current_hits))

        rally_summaries = []
        point_outcomes = []
        score = {"near": 0, "far": 0}
        frame_rally_id = [-1] * frame_count
        frame_shot_count = [0] * frame_count
        frame_last_shot = ["-"] * frame_count
        frame_live_speed = [0.0] * frame_count
        frame_max_speed = [0.0] * frame_count

        for frame_idx in range(frame_count):
            frame_live_speed[frame_idx] = speed_kmh[frame_idx]
            frame_max_speed[frame_idx] = max(speed_kmh[: frame_idx + 1]) if frame_idx > 0 else speed_kmh[0]

        for ridx, (start_f, end_f, hit_list) in enumerate(rallies, start=1):
            last_hitter = None
            last_shot_type = "-"
            rally_max_speed = max(speed_kmh[start_f : end_f + 1]) if end_f >= start_f else 0.0

            for shot_num, hframe in enumerate(hit_list, start=1):
                ball_pt = ball_mini_points[hframe]
                far_pt = far_player_mini[hframe]
                near_pt = near_player_mini[hframe]
                dist_far = self._distance(ball_pt, far_pt) if ball_pt is not None and far_pt is not None else float("inf")
                dist_near = self._distance(ball_pt, near_pt) if ball_pt is not None and near_pt is not None else float("inf")
                hitter = "far" if dist_far <= dist_near else "near"
                last_hitter = hitter

                ball_x = self._bbox_center(ball_detections[hframe][1])[0] if hframe < len(ball_detections) and 1 in ball_detections[hframe] else None
                shot_type = self._infer_shot_type(
                    hitter=hitter,
                    frame_idx=hframe,
                    rally_start=start_f,
                    speed_kmh=speed_kmh[hframe],
                    ball_x=ball_x,
                    near_player_x=near_player_x_img[hframe],
                    far_player_x=far_player_x_img[hframe],
                )
                last_shot_type = shot_type
                hit_events.append(
                    HitEvent(
                        frame=hframe,
                        time_sec=hframe / self.fps,
                        hitter=hitter,
                        shot_type=shot_type,
                        speed_kmh=float(speed_kmh[hframe]),
                    )
                )

                # Fill frame-level stats from hit onward until next hit/end.
                next_frame = hit_list[shot_num] if shot_num < len(hit_list) else end_f + 1
                for fidx in range(hframe, min(next_frame, frame_count)):
                    frame_rally_id[fidx] = ridx
                    frame_shot_count[fidx] = shot_num
                    frame_last_shot[fidx] = shot_type

            winner = None
            if len(hit_list) >= 2 and last_hitter is not None:
                winner = "near" if last_hitter == "far" else "far"
                score[winner] += 1
                point_outcomes.append(
                    {
                        "rally_id": ridx,
                        "start_frame": start_f,
                        "end_frame": end_f,
                        "winner": winner,
                        "shots": len(hit_list),
                    }
                )

            rally_summaries.append(
                {
                    "rally_id": ridx,
                    "start_frame": start_f,
                    "end_frame": end_f,
                    "shots": len(hit_list),
                    "max_speed_kmh": float(rally_max_speed),
                    "last_shot_type": last_shot_type,
                    "winner": winner,
                }
            )

        highlight_segments = self._select_highlight_segments(rally_summaries, frame_count)

        return {
            "hit_events": [event.__dict__ for event in hit_events],
            "rallies": rally_summaries,
            "point_outcomes": point_outcomes,
            "score": score,
            "speed_kmh": speed_kmh,
            "frame_rally_id": frame_rally_id,
            "frame_shot_count": frame_shot_count,
            "frame_last_shot": frame_last_shot,
            "frame_live_speed": frame_live_speed,
            "frame_max_speed": frame_max_speed,
            "highlight_segments": highlight_segments,
        }

    def _select_highlight_segments(self, rally_summaries, frame_count):
        if not rally_summaries:
            return []

        ranked = sorted(
            rally_summaries,
            key=lambda r: (r["shots"] * 12.0 + r["max_speed_kmh"]),
            reverse=True,
        )[:5]

        pre = int(1.0 * self.fps)
        post = int(1.2 * self.fps)
        segments = []
        for idx, rally in enumerate(ranked, start=1):
            start_frame = max(0, rally["start_frame"] - pre)
            end_frame = min(frame_count - 1, rally["end_frame"] + post)
            segments.append(
                {
                    "rank": idx,
                    "rally_id": rally["rally_id"],
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "shots": rally["shots"],
                    "max_speed_kmh": rally["max_speed_kmh"],
                }
            )
        return segments

    def draw_overlay(self, frames, analysis):
        output_frames = []
        score = analysis["score"]
        for i, frame in enumerate(frames):
            panel = frame.copy()
            cv2.rectangle(panel, (20, 40), (520, 230), (18, 18, 18), -1)
            cv2.addWeighted(panel, 0.55, frame, 0.45, 0, frame)

            rally_id = analysis["frame_rally_id"][i]
            shot_count = analysis["frame_shot_count"][i]
            last_shot = analysis["frame_last_shot"][i]
            live_speed = analysis["frame_live_speed"][i]
            max_speed = analysis["frame_max_speed"][i]

            lines = [
                f"Rally: {rally_id if rally_id > 0 else '-'}",
                f"Shots in rally: {shot_count}",
                f"Last shot type: {last_shot}",
                f"Ball speed: {live_speed:.1f} km/h",
                f"Max speed: {max_speed:.1f} km/h",
                f"Score (Near-Far): {score['near']} - {score['far']}",
            ]
            y = 70
            for text in lines:
                cv2.putText(frame, text, (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y += 28

            output_frames.append(frame)
        return output_frames

    def export_events_csv(self, analysis, output_csv_path):
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["frame", "time_sec", "hitter", "shot_type", "speed_kmh"]
        with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            for row in analysis["hit_events"]:
                writer.writerow(row)

    def export_summary_json(self, analysis, output_json_path):
        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "rallies": analysis["rallies"],
            "point_outcomes": analysis["point_outcomes"],
            "score": analysis["score"],
            "highlights": analysis["highlight_segments"],
            "hit_event_count": len(analysis["hit_events"]),
            "max_speed_kmh": max(analysis["speed_kmh"]) if analysis["speed_kmh"] else 0.0,
        }
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(summary, json_file, indent=2)

    def export_highlight_clips(self, frames, analysis, output_dir):
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        if not frames:
            return

        frame_height, frame_width = frames[0].shape[:2]
        for segment in analysis["highlight_segments"]:
            clip_name = (
                f"highlight_{segment['rank']:02d}_"
                f"rally{segment['rally_id']}_shots{segment['shots']}.mp4"
            )
            clip_path = output_root / clip_name
            writer = cv2.VideoWriter(
                str(clip_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (frame_width, frame_height),
            )
            if not writer.isOpened():
                continue
            for frame_idx in range(segment["start_frame"], segment["end_frame"] + 1):
                writer.write(frames[frame_idx])
            writer.release()
