import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2


@dataclass
class Track:
    """Represents a single tracked vehicle across frames."""
    track_id: int
    bbox: np.ndarray          # [x1, y1, x2, y2]
    confidence: float
    class_id: int

    # History for speed estimation
    bbox_history: List[np.ndarray] = field(default_factory=list)
    frame_history: List[int] = field(default_factory=list)
    speed_kmh: float = 0.0
    
    # State
    age: int = 0              # frames since first seen
    hits: int = 0             # total detections matched
    time_since_update: int = 0

    # Behavior flags
    is_stopped: bool = False
    is_wrong_way: bool = False
    sudden_brake_detected: bool = False

    @property
    def center(self) -> np.ndarray:
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        ])

    @property
    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class KalmanBoxTracker:
    """
    Kalman filter for tracking a single bounding box.
    State: [x, y, w, h, vx, vy, vw, vh]
    where x,y = center, w,h = size, v* = velocities
    """

    count = 0

    def __init__(self, bbox: np.ndarray):
        # State transition matrix (constant velocity model)
        self.kf_F = np.array([
            [1,0,0,0, 1,0,0,0],
            [0,1,0,0, 0,1,0,0],
            [0,0,1,0, 0,0,1,0],
            [0,0,0,1, 0,0,0,1],
            [0,0,0,0, 1,0,0,0],
            [0,0,0,0, 0,1,0,0],
            [0,0,0,0, 0,0,1,0],
            [0,0,0,0, 0,0,0,1],
        ], dtype=float)

        # Measurement matrix (we observe x,y,w,h only)
        self.kf_H = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0],
        ], dtype=float)

        # Covariance matrices
        self.kf_R = np.eye(4) * 4.0      # measurement noise
        self.kf_Q = np.eye(8) * 0.01     # process noise
        self.kf_P = np.eye(8) * 10.0     # state covariance

        # Initial state
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w  = x2 - x1
        h  = y2 - y1
        self.kf_x = np.array(
            [[cx], [cy], [w], [h], [0], [0], [0], [0]], dtype=float
        )

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hit_streak = 0
        self.age = 0

    def predict(self) -> np.ndarray:
        """Advance state estimate one step."""
        self.kf_x = self.kf_F @ self.kf_x
        self.kf_P = self.kf_F @ self.kf_P @ self.kf_F.T + self.kf_Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self._state_to_bbox()

    def update(self, bbox: np.ndarray):
        """Update with new measurement."""
        x1, y1, x2, y2 = bbox
        z = np.array([[
            (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
        ]]).T

        # Kalman gain
        S = self.kf_H @ self.kf_P @ self.kf_H.T + self.kf_R
        K = self.kf_P @ self.kf_H.T @ np.linalg.inv(S)

        # Update state
        self.kf_x = self.kf_x + K @ (z - self.kf_H @ self.kf_x)
        self.kf_P = (np.eye(8) - K @ self.kf_H) @ self.kf_P

        self.time_since_update = 0
        self.hit_streak += 1

    def get_velocity(self) -> Tuple[float, float]:
        """Return pixel velocity (vx, vy) from Kalman state."""
        return float(self.kf_x[4]), float(self.kf_x[5])

    def _state_to_bbox(self) -> np.ndarray:
        cx, cy, w, h = (
            self.kf_x[0,0], self.kf_x[1,0],
            self.kf_x[2,0], self.kf_x[3,0]
        )
        return np.array([
            cx - w/2, cy - h/2,
            cx + w/2, cy + h/2
        ])


def iou(bb1: np.ndarray, bb2: np.ndarray) -> float:
    """Compute Intersection over Union between two boxes [x1,y1,x2,y2]."""
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])

    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter == 0:
        return 0.0

    area1 = (bb1[2]-bb1[0]) * (bb1[3]-bb1[1])
    area2 = (bb2[2]-bb2[0]) * (bb2[3]-bb2[1])
    return inter / (area1 + area2 - inter)


def hungarian_match(
    detections: np.ndarray,
    predictions: np.ndarray,
    iou_threshold: float = 0.3
) -> Tuple[List, List, List]:
    """
    Simple IoU-based greedy matching (ByteTrack-style association).
    Returns: matched pairs, unmatched detections, unmatched trackers
    """
    if len(predictions) == 0:
        return [], list(range(len(detections))), []
    if len(detections) == 0:
        return [], [], list(range(len(predictions)))

    # Build IoU matrix
    iou_matrix = np.zeros((len(detections), len(predictions)))
    for d, det in enumerate(detections):
        for t, pred in enumerate(predictions):
            iou_matrix[d, t] = iou(det, pred)

    # Greedy matching (highest IoU first)
    matched = []
    used_det = set()
    used_trk = set()

    # Sort all pairs by IoU descending
    pairs = sorted(
        [(iou_matrix[d,t], d, t)
         for d in range(len(detections))
         for t in range(len(predictions))],
        reverse=True
    )

    for score, d, t in pairs:
        if score < iou_threshold:
            break
        if d not in used_det and t not in used_trk:
            matched.append((d, t))
            used_det.add(d)
            used_trk.add(t)

    unmatched_dets = [d for d in range(len(detections)) if d not in used_det]
    unmatched_trks = [t for t in range(len(predictions)) if t not in used_trk]

    return matched, unmatched_dets, unmatched_trks


class VehicleTracker:
    """
    Multi-object vehicle tracker.
    Combines Kalman filtering with IoU-based association.
    Estimates speed in km/h using pixel displacement + calibration.
    """

    def __init__(
        self,
        fps: float = 25.0,
        pixels_per_meter: float = 8.0,
        max_age: int = 10,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Args:
            fps: Camera frame rate. UA-DETRAC is 25fps.
            pixels_per_meter: Calibration factor.
                              UA-DETRAC overhead cameras ≈ 8 px/m.
                              Adjust per camera if known.
            max_age: Frames to keep a track alive without detection.
            min_hits: Detections needed before track is confirmed.
            iou_threshold: Minimum IoU to match detection to track.
        """
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.trackers: List[KalmanBoxTracker] = []
        self.track_metadata: Dict[int, Track] = {}
        self.frame_count = 0

        # Reset Kalman counter for clean IDs each run
        KalmanBoxTracker.count = 0

    def update(
        self,
        detections: np.ndarray,
        confidences: np.ndarray,
        frame_idx: int = 0
    ) -> List[Track]:
        """
        Update tracker with new detections from YOLO.

        Args:
            detections:  np.ndarray shape (N, 4) — [x1,y1,x2,y2] in pixels
            confidences: np.ndarray shape (N,)
            frame_idx:   current frame number

        Returns:
            List of active Track objects with IDs and speeds
        """
        self.frame_count += 1

        # 1. Predict next position for all existing trackers
        predictions = []
        for trk in self.trackers:
            pred = trk.predict()
            predictions.append(pred)

        # 2. Match detections to predictions
        matched, unmatched_dets, unmatched_trks = hungarian_match(
            detections, predictions, self.iou_threshold
        )

        # 3. Update matched trackers
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(detections[det_idx])

        # 4. Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            new_trk = KalmanBoxTracker(detections[det_idx])
            self.trackers.append(new_trk)

        # 5. Remove dead trackers
        alive = []
        for trk in self.trackers:
            if trk.time_since_update <= self.max_age:
                alive.append(trk)
        self.trackers = alive

        # 6. Build output Track objects
        active_tracks = []
        for i, trk in enumerate(self.trackers):
            if trk.hit_streak < self.min_hits and self.frame_count > self.min_hits:
                continue

            bbox = trk._state_to_bbox()
            track_id = trk.id

            # Get or create metadata
            if track_id not in self.track_metadata:
                self.track_metadata[track_id] = Track(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=confidences[i] if i < len(confidences) else 0.9,
                    class_id=0
                )

            meta = self.track_metadata[track_id]
            meta.bbox = bbox
            meta.age = trk.age
            meta.hits = trk.hit_streak
            meta.time_since_update = trk.time_since_update

            # Update history for speed calculation
            meta.bbox_history.append(bbox.copy())
            meta.frame_history.append(frame_idx)

            # Keep history window to last 10 frames
            if len(meta.bbox_history) > 10:
                meta.bbox_history.pop(0)
                meta.frame_history.pop(0)

            # Estimate speed
            meta.speed_kmh = self._estimate_speed(meta)

            active_tracks.append(meta)

        return active_tracks

    def _estimate_speed(self, track: Track) -> float:
        """
        Estimate vehicle speed in km/h from bounding box displacement.

        Uses center displacement over the last N frames to smooth noise.
        Formula: speed = (pixels / pixels_per_meter) / time * 3.6
        """
        if len(track.bbox_history) < 2:
            return track.speed_kmh  # return last known speed

        # Use displacement over available history window
        old_bbox = track.bbox_history[0]
        new_bbox = track.bbox_history[-1]
        frames_elapsed = max(1, len(track.bbox_history) - 1)

        old_center = np.array([
            (old_bbox[0] + old_bbox[2]) / 2,
            (old_bbox[1] + old_bbox[3]) / 2
        ])
        new_center = np.array([
            (new_bbox[0] + new_bbox[2]) / 2,
            (new_bbox[1] + new_bbox[3]) / 2
        ])

        pixel_displacement = np.linalg.norm(new_center - old_center)
        time_elapsed = frames_elapsed / self.fps

        meters = pixel_displacement / self.pixels_per_meter
        speed_ms = meters / time_elapsed
        speed_kmh = speed_ms * 3.6

        # Smooth with previous estimate (EMA)
        alpha = 0.3
        smoothed = alpha * speed_kmh + (1 - alpha) * track.speed_kmh

        # Clamp to realistic highway range
        return float(np.clip(smoothed, 0, 200))

    def get_track(self, track_id: int) -> Optional[Track]:
        return self.track_metadata.get(track_id)

    def reset(self):
        self.trackers = []
        self.track_metadata = {}
        self.frame_count = 0
        KalmanBoxTracker.count = 0