import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
from collections import deque

from detection.tracker import Track


class BehaviorType(Enum):
    NORMAL            = "NORMAL"
    TAILGATING        = "TAILGATING"
    SUDDEN_BRAKING    = "SUDDEN_BRAKING"
    UNSAFE_LANE_CHANGE = "UNSAFE_LANE_CHANGE"
    STOPPED_VEHICLE   = "STOPPED_VEHICLE"
    WRONG_WAY         = "WRONG_WAY"


@dataclass
class BehaviorEvent:
    """A detected behavior for a specific vehicle."""
    frame_idx:    int
    track_id:     int
    behavior:     BehaviorType
    confidence:   float
    description:  str
    severity:     int   # 1=low, 2=medium, 3=high


class BehaviorClassifier:
    """
    Classifies vehicle behaviors from tracked motion data.

    Five target behaviors:
    1. Tailgating        — following too closely at speed
    2. Sudden braking    — large negative acceleration spike
    3. Unsafe lane change — rapid lateral movement at speed
    4. Stopped vehicle   — speed near zero for multiple frames
    5. Wrong way         — moving opposite to traffic flow

    All thresholds are calibrated for UA-DETRAC overhead cameras
    at 25fps with 8px/m calibration.
    """

    # Tailgating
    TAILGATE_DISTANCE_M   = 10.0   # meters — closer than this is too close
    TAILGATE_SPEED_KMH    = 20.0   # only flag if moving above this speed
    TAILGATE_FRAMES       = 5      # must persist for N frames

    # Sudden braking
    BRAKE_DECEL_THRESHOLD = 15.0   # km/h drop per second
    BRAKE_WINDOW_FRAMES   = 3      # look back N frames

    # Unsafe lane change
    LANE_CHANGE_LATERAL_PX = 20    # pixels of lateral movement per frame
    LANE_CHANGE_SPEED_KMH  = 15.0  # must be moving to count

    # Stopped vehicle
    STOPPED_SPEED_KMH     = 3.0    # below this = stopped
    STOPPED_FRAMES        = 10     # must be stopped for N consecutive frames

    # Wrong way (relative to dominant flow direction)
    WRONG_WAY_FRAMES      = 8      # must persist

    def __init__(
        self,
        fps: float = 25.0,
        pixels_per_meter: float = 8.0,
        image_width: int = 960,
        image_height: int = 540
    ):
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.image_width = image_width
        self.image_height = image_height

        # Per-track state history
        self.speed_history:    Dict[int, deque] = {}
        self.position_history: Dict[int, deque] = {}
        self.stopped_counter:  Dict[int, int]   = {}
        self.wrong_way_counter:Dict[int, int]   = {}
        self.behavior_history: Dict[int, List]  = {}

        # Dominant flow direction (learned from first N frames)
        self.dominant_direction: Optional[np.ndarray] = None
        self.flow_samples: List[np.ndarray] = []

        self.all_events: List[BehaviorEvent] = []

    def update(
        self,
        tracks: List[Track],
        frame_idx: int
    ) -> List[BehaviorEvent]:
        """
        Classify behaviors for all active tracks.

        Args:
            tracks:    Active Track objects from VehicleTracker
            frame_idx: Current frame number

        Returns:
            List of BehaviorEvent objects detected this frame
        """
        frame_events = []

        # Update dominant flow direction from moving vehicles
        self._update_flow_direction(tracks)

        for track in tracks:
            tid = track.track_id
            self._ensure_history(tid)

            # Update histories
            self.speed_history[tid].append(track.speed_kmh)
            if len(track.bbox_history) > 0:
                self.position_history[tid].append(track.center.copy())

            # Need enough history before classifying
            if len(self.speed_history[tid]) < 3:
                continue

            events = []

            # Run each classifier
            e = self._check_stopped(track, frame_idx)
            if e: events.append(e)

            e = self._check_sudden_braking(track, frame_idx)
            if e: events.append(e)

            e = self._check_wrong_way(track, frame_idx)
            if e: events.append(e)

            e = self._check_unsafe_lane_change(track, frame_idx)
            if e: events.append(e)

            # Tailgating checked at pair level — handled by pipeline
            # (needs two tracks, so done in pipeline.py)

            frame_events.extend(events)
            self.all_events.extend(events)

        return frame_events

    def check_tailgating(
        self,
        track_a: Track,
        track_b: Track,
        frame_idx: int
    ) -> Optional[BehaviorEvent]:
        """
        Check if track_a is tailgating track_b.
        Called from pipeline.py for each vehicle pair.
        """
        dist_pixels = np.linalg.norm(track_a.center - track_b.center)
        dist_meters = dist_pixels / self.pixels_per_meter
        tid = track_a.track_id
        self._ensure_history(tid)

        # Must be close AND moving
        if (dist_meters < self.TAILGATE_DISTANCE_M and
                track_a.speed_kmh > self.TAILGATE_SPEED_KMH):

            

            # Check persistence
            key = f"tailgate_{tid}_{track_b.track_id}"
            if key not in self.stopped_counter:
                self.stopped_counter[key] = 0
            self.stopped_counter[key] += 1

            if self.stopped_counter[key] >= self.TAILGATE_FRAMES:
                confidence = min(0.95,
                    0.6 + (self.TAILGATE_DISTANCE_M - dist_meters) * 0.03
                )
                event = BehaviorEvent(
                    frame_idx=frame_idx,
                    track_id=tid,
                    behavior=BehaviorType.TAILGATING,
                    confidence=confidence,
                    description=(
                        f"🚗 TAILGATING | Vehicle {tid} → {track_b.track_id} | "
                        f"Gap: {dist_meters:.1f}m | "
                        f"Speed: {track_a.speed_kmh:.0f} km/h"
                    ),
                    severity=2
                )
                return event
        else:
            # Reset counter if condition no longer met
            key = f"tailgate_{tid}_{track_b.track_id}"
            if key in self.stopped_counter:
                self.stopped_counter[key] = 0

        return None

    def _check_stopped(
        self, track: Track, frame_idx: int
    ) -> Optional[BehaviorEvent]:
        """Detect vehicles stopped on the roadway."""
        tid = track.track_id

        if track.speed_kmh < self.STOPPED_SPEED_KMH:
            self.stopped_counter[f"stop_{tid}"] = \
                self.stopped_counter.get(f"stop_{tid}", 0) + 1
        else:
            self.stopped_counter[f"stop_{tid}"] = 0

        count = self.stopped_counter.get(f"stop_{tid}", 0)

        if count == self.STOPPED_FRAMES:  # fire once at threshold
            return BehaviorEvent(
                frame_idx=frame_idx,
                track_id=tid,
                behavior=BehaviorType.STOPPED_VEHICLE,
                confidence=0.88,
                description=(
                    f"🛑 STOPPED VEHICLE | ID:{tid} | "
                    f"Stationary for {count} frames"
                ),
                severity=2
            )
        return None

    def _check_sudden_braking(
        self, track: Track, frame_idx: int
    ) -> Optional[BehaviorEvent]:
        """Detect sudden deceleration events."""
        tid = track.track_id
        speeds = list(self.speed_history[tid])

        if len(speeds) < self.BRAKE_WINDOW_FRAMES:
            return None

        recent  = speeds[-1]
        earlier = speeds[-self.BRAKE_WINDOW_FRAMES]
        time_s  = self.BRAKE_WINDOW_FRAMES / self.fps

        # Deceleration in km/h per second
        decel = (earlier - recent) / time_s

        if decel > self.BRAKE_DECEL_THRESHOLD and earlier > 10.0:
            confidence = min(0.95, 0.5 + decel / 100)
            return BehaviorEvent(
                frame_idx=frame_idx,
                track_id=tid,
                behavior=BehaviorType.SUDDEN_BRAKING,
                confidence=confidence,
                description=(
                    f"🔴 SUDDEN BRAKING | ID:{tid} | "
                    f"{earlier:.0f} → {recent:.0f} km/h | "
                    f"Decel: {decel:.0f} km/h/s"
                ),
                severity=3
            )
        return None

    def _check_unsafe_lane_change(
        self, track: Track, frame_idx: int
    ) -> Optional[BehaviorEvent]:
        """Detect rapid lateral movement while at speed."""
        tid = track.track_id
        positions = list(self.position_history[tid])

        if len(positions) < 2:
            return None

        if track.speed_kmh < self.LANE_CHANGE_SPEED_KMH:
            return None

        # Lateral displacement (x-axis in overhead camera)
        lateral_px = abs(positions[-1][0] - positions[-2][0])

        if lateral_px > self.LANE_CHANGE_LATERAL_PX:
            lateral_m = lateral_px / self.pixels_per_meter
            confidence = min(0.9, 0.5 + lateral_px / 100)
            return BehaviorEvent(
                frame_idx=frame_idx,
                track_id=tid,
                behavior=BehaviorType.UNSAFE_LANE_CHANGE,
                confidence=confidence,
                description=(
                    f"↔️  UNSAFE LANE CHANGE | ID:{tid} | "
                    f"Lateral: {lateral_m:.1f}m | "
                    f"Speed: {track.speed_kmh:.0f} km/h"
                ),
                severity=2
            )
        return None

    def _check_wrong_way(
        self, track: Track, frame_idx: int
    ) -> Optional[BehaviorEvent]:
        """Detect vehicles moving against dominant traffic flow."""
        if self.dominant_direction is None:
            return None

        tid = track.track_id
        positions = list(self.position_history[tid])

        if len(positions) < 2 or track.speed_kmh < 5.0:
            return None

        # Vehicle's current movement vector
        movement = positions[-1] - positions[-2]
        if np.linalg.norm(movement) < 0.5:
            return None

        movement_norm = movement / np.linalg.norm(movement)

        # Dot product — negative means opposing direction
        alignment = np.dot(movement_norm, self.dominant_direction)

        if alignment < -0.5:  # more than 120° from dominant flow
            self.wrong_way_counter[tid] = \
                self.wrong_way_counter.get(tid, 0) + 1
        else:
            self.wrong_way_counter[tid] = 0

        count = self.wrong_way_counter.get(tid, 0)

        if count == self.WRONG_WAY_FRAMES:
            return BehaviorEvent(
                frame_idx=frame_idx,
                track_id=tid,
                behavior=BehaviorType.WRONG_WAY,
                confidence=0.85,
                description=(
                    f"⛔ WRONG WAY | ID:{tid} | "
                    f"Speed: {track.speed_kmh:.0f} km/h"
                ),
                severity=3
            )
        return None

    def _update_flow_direction(self, tracks: List[Track]):
        """
        Learn dominant traffic flow direction from moving vehicles.
        Averages velocity vectors from the first 50 moving vehicles seen.
        """
        if self.dominant_direction is not None:
            return  # already learned

        for track in tracks:
            if (track.speed_kmh > 10 and
                    len(track.bbox_history) >= 2):
                old_c = np.array([
                    (track.bbox_history[-2][0] + track.bbox_history[-2][2]) / 2,
                    (track.bbox_history[-2][1] + track.bbox_history[-2][3]) / 2
                ])
                new_c = track.center
                vec = new_c - old_c
                if np.linalg.norm(vec) > 0.5:
                    self.flow_samples.append(vec / np.linalg.norm(vec))

        if len(self.flow_samples) >= 20:
            avg = np.mean(self.flow_samples, axis=0)
            self.dominant_direction = avg / np.linalg.norm(avg)

    def _ensure_history(self, tid: int):
        if tid not in self.speed_history:
            self.speed_history[tid]    = deque(maxlen=30)
            self.position_history[tid] = deque(maxlen=30)
            self.behavior_history[tid] = []

    def get_summary(self) -> Dict:
        by_type = {}
        for e in self.all_events:
            k = e.behavior.value
            by_type[k] = by_type.get(k, 0) + 1
        return {
            "total_behaviors": len(self.all_events),
            "by_type": by_type
        }