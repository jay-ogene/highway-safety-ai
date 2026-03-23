import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from enum import Enum

from detection.tracker import Track


class AlertLevel(Enum):
    SAFE     = "SAFE"
    WARNING  = "WARNING"    # TTC < 3.0s
    HIGH     = "HIGH"       # TTC < 2.5s
    CRITICAL = "CRITICAL"   # TTC < 1.5s


@dataclass
class NearMissEvent:
    """Represents a detected near-miss between two vehicles."""
    frame_idx: int
    vehicle_a_id: int
    vehicle_b_id: int
    ttc_seconds: float
    distance_meters: float
    closing_speed_kmh: float
    alert_level: AlertLevel
    description: str


@dataclass
class VehiclePair:
    """Tracks the relationship between two vehicles over time."""
    id_a: int
    id_b: int
    ttc_history: List[float]
    alert_level: AlertLevel = AlertLevel.SAFE

    @property
    def pair_key(self) -> str:
        return f"{min(self.id_a, self.id_b)}_{max(self.id_a, self.id_b)}"


class NearMissDetector:
    """
    Detects near-miss incidents between tracked vehicles.

    Uses Time-To-Collision (TTC) as the primary metric:
        TTC = distance / closing_speed

    Alert thresholds (from traffic safety research):
        TTC < 3.0s  → WARNING
        TTC < 2.5s  → HIGH
        TTC < 1.5s  → CRITICAL

    Reference: Hayward (1972) — TTC as a measure of traffic conflict severity
    """

    # TTC thresholds in seconds
    TTC_WARNING  = 3.0
    TTC_HIGH     = 2.5
    TTC_CRITICAL = 1.5

    # Maximum distance to consider vehicles as a pair (meters)
    MAX_INTERACTION_DISTANCE = 50.0

    def __init__(self, pixels_per_meter: float = 8.0, fps: float = 25.0):
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        self.pair_history: Dict[str, VehiclePair] = {}
        self.events: List[NearMissEvent] = []

    def update(
        self,
        tracks: List[Track],
        frame_idx: int
    ) -> List[NearMissEvent]:
        """
        Analyze all vehicle pairs for near-miss conditions.

        Args:
            tracks:    Active tracks from VehicleTracker.update()
            frame_idx: Current frame number

        Returns:
            List of NearMissEvent objects for this frame
        """
        frame_events = []

        if len(tracks) < 2:
            return frame_events

        # Check every pair of vehicles
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                event = self._analyze_pair(
                    tracks[i], tracks[j], frame_idx
                )
                if event is not None:
                    frame_events.append(event)
                    self.events.append(event)

        return frame_events

    def _analyze_pair(
        self,
        track_a: Track,
        track_b: Track,
        frame_idx: int
    ) -> Optional[NearMissEvent]:
        """Analyze a single vehicle pair for collision risk."""

        # Calculate pixel distance between vehicle centers
        dist_pixels = np.linalg.norm(track_a.center - track_b.center)
        dist_meters = dist_pixels / self.pixels_per_meter

        # Skip pairs that are too far apart
        if dist_meters > self.MAX_INTERACTION_DISTANCE:
            return None

        # Calculate closing speed (positive = approaching)
        closing_speed_kmh = self._closing_speed(track_a, track_b)
        closing_speed_ms  = closing_speed_kmh / 3.6

        # TTC only meaningful when vehicles are approaching
        if closing_speed_ms <= 0:
            return None

        ttc = dist_meters / closing_speed_ms

        # Determine alert level
        alert = self._alert_level(ttc)

        # Only report WARNING and above
        if alert == AlertLevel.SAFE:
            return None

        # Update pair history
        pair_key = f"{min(track_a.track_id, track_b.track_id)}_" \
                   f"{max(track_a.track_id, track_b.track_id)}"

        if pair_key not in self.pair_history:
            self.pair_history[pair_key] = VehiclePair(
                id_a=track_a.track_id,
                id_b=track_b.track_id,
                ttc_history=[]
            )

        pair = self.pair_history[pair_key]
        pair.ttc_history.append(ttc)
        pair.alert_level = alert

        description = self._describe_event(
            track_a, track_b, ttc, dist_meters,
            closing_speed_kmh, alert
        )

        return NearMissEvent(
            frame_idx=frame_idx,
            vehicle_a_id=track_a.track_id,
            vehicle_b_id=track_b.track_id,
            ttc_seconds=round(ttc, 2),
            distance_meters=round(dist_meters, 1),
            closing_speed_kmh=round(closing_speed_kmh, 1),
            alert_level=alert,
            description=description
        )

    def _closing_speed(self, track_a: Track, track_b: Track) -> float:
        """
        Estimate closing speed between two vehicles in km/h.

        Uses the rate of change in distance between centers
        over the bbox history window.
        """
        # Need at least 2 frames of history for both tracks
        if (len(track_a.bbox_history) < 2 or
                len(track_b.bbox_history) < 2):
            # Fall back to difference in speeds along axis
            speed_diff = abs(track_a.speed_kmh - track_b.speed_kmh)
            return speed_diff * 0.5  # conservative estimate

        # Current distance
        center_a_now = np.array([
            (track_a.bbox_history[-1][0] + track_a.bbox_history[-1][2]) / 2,
            (track_a.bbox_history[-1][1] + track_a.bbox_history[-1][3]) / 2
        ])
        center_b_now = np.array([
            (track_b.bbox_history[-1][0] + track_b.bbox_history[-1][2]) / 2,
            (track_b.bbox_history[-1][1] + track_b.bbox_history[-1][3]) / 2
        ])
        dist_now = np.linalg.norm(center_a_now - center_b_now)

        # Previous distance
        center_a_prev = np.array([
            (track_a.bbox_history[-2][0] + track_a.bbox_history[-2][2]) / 2,
            (track_a.bbox_history[-2][1] + track_a.bbox_history[-2][3]) / 2
        ])
        center_b_prev = np.array([
            (track_b.bbox_history[-2][0] + track_b.bbox_history[-2][2]) / 2,
            (track_b.bbox_history[-2][1] + track_b.bbox_history[-2][3]) / 2
        ])
        dist_prev = np.linalg.norm(center_a_prev - center_b_prev)

        # Closing speed: positive means approaching
        delta_pixels = dist_prev - dist_now
        delta_meters = delta_pixels / self.pixels_per_meter
        closing_ms   = delta_meters * self.fps
        closing_kmh  = closing_ms * 3.6

        return max(0.0, closing_kmh)

    def _alert_level(self, ttc: float) -> AlertLevel:
        if ttc < self.TTC_CRITICAL:
            return AlertLevel.CRITICAL
        elif ttc < self.TTC_HIGH:
            return AlertLevel.HIGH
        elif ttc < self.TTC_WARNING:
            return AlertLevel.WARNING
        return AlertLevel.SAFE

    def _describe_event(
        self,
        track_a: Track,
        track_b: Track,
        ttc: float,
        dist_meters: float,
        closing_speed_kmh: float,
        alert: AlertLevel
    ) -> str:
        prefix = {
            AlertLevel.WARNING:  "⚠️  WARNING",
            AlertLevel.HIGH:     "🟠 HIGH RISK",
            AlertLevel.CRITICAL: "🔴 NEAR MISS"
        }[alert]

        return (
            f"{prefix} | "
            f"Vehicles {track_a.track_id} & {track_b.track_id} | "
            f"TTC: {ttc:.1f}s | "
            f"Distance: {dist_meters:.1f}m | "
            f"Closing: {closing_speed_kmh:.0f} km/h"
        )

    def get_summary(self) -> Dict:
        """Return summary statistics for all detected events."""
        if not self.events:
            return {"total_events": 0}

        by_level = {}
        for level in AlertLevel:
            count = sum(1 for e in self.events if e.alert_level == level)
            if count > 0:
                by_level[level.value] = count

        return {
            "total_events":    len(self.events),
            "by_level":        by_level,
            "min_ttc":         min(e.ttc_seconds for e in self.events),
            "avg_ttc":         np.mean([e.ttc_seconds for e in self.events]),
            "pairs_involved":  len(self.pair_history)
        }