import numpy as np
import cv2
import Constants
from utils import (
    convert_meters_to_pixel_distance
)


class MiniCourt:
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 450
        self.buffer = 50
        self.padding_court = 20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

    @staticmethod
    def _to_points(flat_keypoints):
        points = np.array(flat_keypoints, dtype=np.float32)
        return points.reshape(-1, 2)

    def _compute_homography(self, court_keypoints):
        if (
            court_keypoints is None
            or len(court_keypoints) != len(self.reference_key_points_model_order)
        ):
            return None

        src_pts = self._to_points(court_keypoints)
        dst_pts = self._to_points(self.reference_key_points_model_order)
        homography, _ = cv2.findHomography(src_pts, dst_pts, method=0)
        return homography

    @staticmethod
    def _project_point(point, homography):
        if homography is None:
            return None
        point_arr = np.array([[[point[0], point[1]]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(point_arr, homography)[0][0]
        return int(projected[0]), int(projected[1])

    @staticmethod
    def _distance(p1, p2):
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    def _is_inside_mini_court(self, point, margin=8):
        if point is None:
            return False
        x, y = point
        return (
            self.court_start_x - margin <= x <= self.court_end_x + margin
            and self.court_start_y - margin <= y <= self.court_end_y + margin
        )

    def _clamp_to_mini_court(self, point):
        if point is None:
            return None
        x = min(max(point[0], self.court_start_x), self.court_end_x)
        y = min(max(point[1], self.court_start_y), self.court_end_y)
        return x, y

    def _stabilize_point(self, projected, prev_point, max_jump, alpha):
        if projected is None:
            return prev_point
        if not self._is_inside_mini_court(projected, margin=60):
            return prev_point
        projected = self._clamp_to_mini_court(projected)
        if prev_point is None:
            return projected
        if self._distance(projected, prev_point) > max_jump:
            return prev_point
        smoothed_x = int(alpha * projected[0] + (1 - alpha) * prev_point[0])
        smoothed_y = int(alpha * projected[1] + (1 - alpha) * prev_point[1])
        return smoothed_x, smoothed_y

    def _stabilize_ball_point(self, projected, prev_point):
        if projected is None:
            return prev_point
        if not self._is_inside_mini_court(projected, margin=100):
            return prev_point
        projected = self._clamp_to_mini_court(projected)
        if prev_point is None:
            return projected

        # Preserve motion while preventing implausible teleports.
        alpha = 0.8
        target_x = alpha * projected[0] + (1 - alpha) * prev_point[0]
        target_y = alpha * projected[1] + (1 - alpha) * prev_point[1]
        dx = target_x - prev_point[0]
        dy = target_y - prev_point[1]
        step = float(np.hypot(dx, dy))
        max_step = max(24, int(self.court_drawing_width * 0.22))
        if step > max_step and step > 0:
            scale = max_step / step
            target_x = prev_point[0] + dx * scale
            target_y = prev_point[1] + dy * scale
        return int(target_x), int(target_y)

    @staticmethod
    def _assign_by_cost(candidates, prev_a, prev_b):
        if len(candidates) < 2:
            return None
        c0, c1 = candidates[0], candidates[1]
        cost_same = MiniCourt._distance(c0, prev_a) + MiniCourt._distance(c1, prev_b)
        cost_swap = MiniCourt._distance(c0, prev_b) + MiniCourt._distance(c1, prev_a)
        if cost_same <= cost_swap:
            return c0, c1
        return c1, c0

    def _assign_player_roles(self, projected_candidates, prev_far, prev_near):
        if not projected_candidates:
            return prev_far, prev_near

        # Keep up to two most separated points vertically to represent far/near players.
        projected_candidates = sorted(projected_candidates, key=lambda p: p[1])
        if len(projected_candidates) >= 2:
            candidates = [projected_candidates[0], projected_candidates[-1]]
        else:
            candidates = [projected_candidates[0]]

        if len(candidates) == 1:
            only = candidates[0]
            if prev_far is None and prev_near is None:
                mid_y = (self.court_start_y + self.court_end_y) / 2.0
                if only[1] <= mid_y:
                    return only, prev_near
                return prev_far, only
            if prev_far is None:
                return only, prev_near
            if prev_near is None:
                return prev_far, only
            if self._distance(only, prev_far) <= self._distance(only, prev_near):
                return only, prev_near
            return prev_far, only

        far_cand, near_cand = candidates[0], candidates[1]
        if prev_far is not None and prev_near is not None:
            assigned = self._assign_by_cost([far_cand, near_cand], prev_far, prev_near)
            if assigned is not None:
                return assigned[0], assigned[1]
        return far_cand, near_cand

    def convert_meters_to_pixels(self, meters):
        return meters * self.pixels_per_meter

    def set_canvas_background_box_position(self, frame):
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x
        self.court_drawing_height = self.court_end_y - self.court_start_y
        court_length_m = Constants.HALF_COURT_LINE_HEIGHT * 2
        court_width_m = Constants.DOUBLE_LINE_WIDTH
        self.pixels_per_meter = min(
            self.court_drawing_width / court_width_m,
            self.court_drawing_height / court_length_m,
        )

    def set_court_drawing_key_points(self):
        """
        Generates 2D points of key positions on a tennis court within the mini rectangle.
        """
        drawing_key_points = [0] * 28

        # Court corner points
        court_pixel_width = self.convert_meters_to_pixels(Constants.DOUBLE_LINE_WIDTH)
        court_pixel_height = self.convert_meters_to_pixels(Constants.HALF_COURT_LINE_HEIGHT * 2)
        left = self.court_start_x + int((self.court_drawing_width - court_pixel_width) / 2)
        top = self.court_start_y + int((self.court_drawing_height - court_pixel_height) / 2)
        right = left + court_pixel_width
        bottom = top + court_pixel_height

        mid_x = (left + right) // 2
        no_mans_land_height = self.convert_meters_to_pixels(Constants.NO_MANS_LAND_HEIGTH)

        ally_offset = self.convert_meters_to_pixels(Constants.DOUBLE_ALLY_DIFFERENCE)

        # Drawing key points
        drawing_key_points[0] = left                # pt 0 (top-left corner)
        drawing_key_points[1] = bottom

        drawing_key_points[2] = right               # pt 1 (top-right corner)
        drawing_key_points[3] = bottom

        drawing_key_points[4] = left                # pt 2 (bottom-left corner)
        drawing_key_points[5] = top

        drawing_key_points[6] = right               # pt 3 (bottom-right corner)
        drawing_key_points[7] = top

        drawing_key_points[8] = mid_x               # pt 4 (center-bottom)
        drawing_key_points[9] = bottom

        drawing_key_points[10] = mid_x              # pt 5 (center-top)
        drawing_key_points[11] = top

        drawing_key_points[12] = left + ally_offset                     # pt 6 (singles-left)
        drawing_key_points[13] = bottom

        drawing_key_points[14] = left + ally_offset                     # pt 7 (singles-left)
        drawing_key_points[15] = top

        drawing_key_points[16] = right - ally_offset                    # pt 8 (singles-right)
        drawing_key_points[17] = bottom

        drawing_key_points[18] = right - ally_offset                    # pt 9 (singles-right)
        drawing_key_points[19] = top

        drawing_key_points[20] = left                                   # pt 10 (service line bottom)
        drawing_key_points[21] = bottom - no_mans_land_height

        drawing_key_points[22] = right                                  # pt 11 (service line bottom)
        drawing_key_points[23] = bottom - no_mans_land_height

        drawing_key_points[24] = left                                   # pt 12 (service line top)
        drawing_key_points[25] = top + no_mans_land_height

        drawing_key_points[26] = right                                  # pt 13 (service line top)
        drawing_key_points[27] = top + no_mans_land_height

        self.drawing_key_points = drawing_key_points

        # Model order from court detector:
        # 0 TL outer, 1 TR outer, 2 BL outer, 3 BR outer,
        # 4 TL singles, 5 BL singles, 6 TR singles, 7 BR singles,
        # 8 TL service, 9 TR service, 10 BL service, 11 BR service,
        # 12 top T, 13 bottom T.
        top_service_y = top + no_mans_land_height
        bottom_service_y = bottom - no_mans_land_height
        left_singles_x = left + ally_offset
        right_singles_x = right - ally_offset

        self.reference_key_points_model_order = [
            left, top,
            right, top,
            left, bottom,
            right, bottom,
            left_singles_x, top,
            left_singles_x, bottom,
            right_singles_x, top,
            right_singles_x, bottom,
            left_singles_x, top_service_y,
            right_singles_x, top_service_y,
            left_singles_x, bottom_service_y,
            right_singles_x, bottom_service_y,
            mid_x, top_service_y,
            mid_x, bottom_service_y,
        ]

    def set_court_lines(self):
        """
        Pairs of points (indices in drawing_key_points) to be connected as lines.
        """
        self.lines = [
            (0, 1), (0, 2), (1, 3), (2, 3),   # outer rectangle
            (4, 5),                            # center line
            (10, 11), (12, 13),               # service lines
            (6, 7), (8, 9)                    # singles sidelines
        ]

    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        return out

    def draw_court(self, frame):
        # Draw court lines
        for start, end in self.lines:
            x1 = int(self.drawing_key_points[start * 2])
            y1 = int(self.drawing_key_points[start * 2 + 1])
            x2 = int(self.drawing_key_points[end * 2])
            y2 = int(self.drawing_key_points[end * 2 + 1])
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame

    def draw_mini_court(self, frames, player_detections=None, ball_detections=None, court_keypoints=None):
        homography = self._compute_homography(court_keypoints)
        prev_far_player_point = None
        prev_near_player_point = None
        prev_ball_point = None
        ball_trail = []
        max_player_jump = max(18, int(self.court_drawing_width * 0.18))

        output_frames = []
        for idx, frame in enumerate(frames):
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)

            if homography is not None:
                if player_detections is not None and idx < len(player_detections):
                    player_dict = player_detections[idx]
                    projected_candidates = []
                    for _, bbox in player_dict.items():
                        x1, y1, x2, y2 = bbox
                        player_anchor = ((x1 + x2) / 2.0, y1 + 0.82 * (y2 - y1))
                        projected = self._project_point(player_anchor, homography)
                        if projected is not None and self._is_inside_mini_court(projected, margin=80):
                            projected_candidates.append(self._clamp_to_mini_court(projected))

                    far_target, near_target = self._assign_player_roles(
                        projected_candidates,
                        prev_far_player_point,
                        prev_near_player_point,
                    )
                    prev_far_player_point = self._stabilize_point(
                        far_target, prev_far_player_point, max_player_jump, alpha=0.30
                    )
                    prev_near_player_point = self._stabilize_point(
                        near_target, prev_near_player_point, max_player_jump, alpha=0.30
                    )
                    if prev_far_player_point is not None:
                        cv2.circle(frame, prev_far_player_point, 5, (255, 220, 0), -1)
                    if prev_near_player_point is not None:
                        cv2.circle(frame, prev_near_player_point, 5, (0, 255, 255), -1)

                if ball_detections is not None and idx < len(ball_detections):
                    ball_dict = ball_detections[idx]
                    if 1 in ball_dict:
                        x1, y1, x2, y2 = ball_dict[1]
                        ball_ground_anchor = ((x1 + x2) / 2.0, y2)
                        projected = self._project_point(ball_ground_anchor, homography)
                        prev_ball_point = self._stabilize_ball_point(projected, prev_ball_point)
                    if prev_ball_point is not None:
                        ball_trail.append(prev_ball_point)
                        if len(ball_trail) > 8:
                            ball_trail = ball_trail[-8:]
                        for i in range(1, len(ball_trail)):
                            cv2.line(frame, ball_trail[i - 1], ball_trail[i], (0, 140, 255), 1)
                        cv2.circle(frame, prev_ball_point, 6, (0, 0, 255), -1)
                        cv2.circle(frame, prev_ball_point, 8, (0, 0, 0), 1)

            output_frames.append(frame)
        return output_frames
