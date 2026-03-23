import cv2
import numpy as np
import os
import glob
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

from detection.tracker import VehicleTracker, Track
from detection.near_miss import NearMissDetector, NearMissEvent, AlertLevel
from detection.behavior_classifier import (
    BehaviorClassifier, BehaviorEvent, BehaviorType
)


# ── Overlay colors (BGR for OpenCV) ──────────────────────────────
COLOR = {
    "box_default":  (200, 200, 200),
    "box_warning":  (0, 165, 255),
    "box_high":     (0, 100, 255),
    "box_critical": (0, 0, 255),
    "speed_text":   (255, 255, 255),
    "id_text":      (100, 255, 100),
    "warning":      (0, 165, 255),
    "critical":     (0, 0, 255),
    "safe":         (100, 255, 100),
    "background":   (0, 0, 0),
    "stopped":      (0, 0, 220),
    "wrong_way":    (0, 0, 255),
    "braking":      (0, 100, 255),
    "lane_change":  (255, 165, 0),
    "tailgate":     (255, 100, 0),
}

@dataclass
class FrameResult:
    """All detections and events for a single frame."""
    frame_idx:       int
    tracks:          List[Track]
    near_miss_events: List[NearMissEvent]
    behavior_events:  List[BehaviorEvent]
    annotated_frame:  Optional[np.ndarray] = None


class HighwaySafetyPipeline:
    """
    Full end-to-end highway safety pipeline.

    Flow per frame:
      1. Load frame from UA-DETRAC sequence
      2. YOLO detects vehicles → bounding boxes
      3. Tracker assigns persistent IDs + estimates speed
      4. NearMissDetector calculates TTC for all pairs
      5. BehaviorClassifier flags incident behaviors
      6. Overlay all results onto frame
      7. Save annotated frame
    """

    def __init__(
        self,
        model_path: str,
        fps: float = 25.0,
        pixels_per_meter: float = 8.0,
        confidence_threshold: float = 0.4,
        output_dir: str = "output/pipeline"
    ):
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.confidence_threshold = confidence_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load YOLO model
        print(f"Loading model from {model_path}...")
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        print("Model loaded ✅")

        # Initialize components
        self.tracker = VehicleTracker(
            fps=fps,
            pixels_per_meter=pixels_per_meter,
            max_age=10,
            min_hits=2,
            iou_threshold=0.3
        )
        self.near_miss = NearMissDetector(
            pixels_per_meter=pixels_per_meter,
            fps=fps
        )
        self.classifier = BehaviorClassifier(
            fps=fps,
            pixels_per_meter=pixels_per_meter
        )

        # Stats
        self.total_frames    = 0
        self.total_incidents = 0
        self.frame_results: List[FrameResult] = []

        # Deduplication — avoid repeating same alert every frame
        self.active_alerts: dict = {}

    def run_sequence(
        self,
        sequence_dir: str,
        sequence_id: str = "MVI_20011",
        max_frames: Optional[int] = None,
        save_video: bool = True
    ) -> List[FrameResult]:
        """
        Process a full UA-DETRAC sequence.

        Args:
            sequence_dir: Path to UA-DETRAC images/val directory
            sequence_id:  MVI prefix to process e.g. 'MVI_20011'
            max_frames:   Limit frames for testing (None = all)
            save_video:   Whether to assemble output video

        Returns:
            List of FrameResult for each processed frame
        """
        # Find all frames for this sequence
        pattern = os.path.join(sequence_dir, f"{sequence_id}_img*.jpg")
        frame_paths = sorted(glob.glob(pattern))

        if not frame_paths:
            raise FileNotFoundError(
                f"No frames found for {sequence_id} in {sequence_dir}"
            )

        if max_frames:
            frame_paths = frame_paths[:max_frames]

        print(f"\nProcessing {sequence_id}")
        print(f"Frames: {len(frame_paths)}")
        print(f"Output: {self.output_dir}")
        print("-" * 50)

        results = []
        for idx, frame_path in enumerate(frame_paths):
            result = self.process_frame(
                frame_path=frame_path,
                frame_idx=idx,
                save=True
            )
            results.append(result)
            self.frame_results.append(result)

            # Progress every 25 frames
            if idx % 25 == 0:
                n_inc = len(result.near_miss_events) + \
                        len(result.behavior_events)
                print(
                    f"Frame {idx:4d}/{len(frame_paths)} | "
                    f"Tracks: {len(result.tracks):2d} | "
                    f"Incidents: {n_inc}"
                )

        print(f"\nSequence complete: {len(results)} frames processed")

        if save_video:
            self._assemble_video(sequence_id)

        self._print_summary(sequence_id)
        return results

    def process_frame(
        self,
        frame_path: str,
        frame_idx: int,
        save: bool = True
    ) -> FrameResult:
        """Process a single frame through the full pipeline."""
        self.total_frames += 1

        # 1. Load frame
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Could not load frame: {frame_path}")

        h, w = frame.shape[:2]

        # 2. YOLO detection
        yolo_results = self.model(
            frame,
            conf=self.confidence_threshold,
            verbose=False
        )[0]

        detections = []
        confidences = []

        if yolo_results.boxes is not None and len(yolo_results.boxes):
            boxes = yolo_results.boxes.xyxy.cpu().numpy()
            confs = yolo_results.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                detections.append(box)
                confidences.append(conf)

        detections  = np.array(detections)  if detections  else np.empty((0,4))
        confidences = np.array(confidences) if confidences else np.empty((0,))

        # 3. Update tracker
        tracks = self.tracker.update(detections, confidences, frame_idx)

        # 4. Near-miss detection
        nm_events = self.near_miss.update(tracks, frame_idx)

        # 5. Behavior classification (single vehicles)
        beh_events = self.classifier.update(tracks, frame_idx)

        # 5b. Tailgating check (vehicle pairs)
        for i in range(len(tracks)):
            for j in range(i+1, len(tracks)):
                e = self.classifier.check_tailgating(
                    tracks[i], tracks[j], frame_idx
                )
                if e:
                    beh_events.append(e)

        # 6. Draw overlay
        annotated = self._draw_overlay(
            frame.copy(), tracks, nm_events, beh_events, frame_idx
        )

        # 7. Save frame
        if save:
            out_path = self.output_dir / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(out_path), annotated)

        self.total_incidents += len(nm_events) + len(beh_events)

        return FrameResult(
            frame_idx=frame_idx,
            tracks=tracks,
            near_miss_events=nm_events,
            behavior_events=beh_events,
            annotated_frame=annotated
        )

    def _draw_overlay(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        nm_events: List[NearMissEvent],
        beh_events: List[BehaviorEvent],
        frame_idx: int
    ) -> np.ndarray:
        """Draw all annotations onto the frame."""

        # Build lookup: which tracks have active alerts
        alert_tracks = {}
        for e in nm_events:
            alert_tracks[e.vehicle_a_id] = e.alert_level
            alert_tracks[e.vehicle_b_id] = e.alert_level
        for e in beh_events:
            if e.behavior in [
                BehaviorType.SUDDEN_BRAKING,
                BehaviorType.WRONG_WAY
            ]:
                if e.track_id not in alert_tracks:
                    alert_tracks[e.track_id] = AlertLevel.HIGH

        # Draw each tracked vehicle
        for track in tracks:
            self._draw_vehicle(frame, track, alert_tracks)

        # Draw incident alerts panel (top of frame)
        self._draw_alert_panel(frame, nm_events, beh_events)

        # Draw frame stats (bottom left)
        self._draw_stats(frame, tracks, frame_idx)

        return frame

    def _draw_vehicle(
        self,
        frame: np.ndarray,
        track: Track,
        alert_tracks: dict
    ):
        """Draw bounding box, ID, and speed for one vehicle."""
        x1, y1, x2, y2 = track.bbox.astype(int)

        # Box color based on alert level
        alert = alert_tracks.get(track.track_id)
        if alert == AlertLevel.CRITICAL:
            box_color = COLOR["box_critical"]
            thickness = 3
        elif alert == AlertLevel.HIGH:
            box_color = COLOR["box_high"]
            thickness = 2
        elif alert == AlertLevel.WARNING:
            box_color = COLOR["box_warning"]
            thickness = 2
        else:
            box_color = COLOR["box_default"]
            thickness = 1

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

        # Label background
        label = f"ID:{track.track_id} {track.speed_kmh:.0f}km/h"
        (lw, lh), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        label_y = max(y1 - 4, lh + 4)
        cv2.rectangle(
            frame,
            (x1, label_y - lh - 4),
            (x1 + lw + 4, label_y + 2),
            box_color, -1
        )

        # Label text
        cv2.putText(
            frame, label,
            (x1 + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            COLOR["speed_text"], 1, cv2.LINE_AA
        )

    def _draw_alert_panel(
        self,
        frame: np.ndarray,
        nm_events: List[NearMissEvent],
        beh_events: List[BehaviorEvent]
    ):
        """Draw active incident alerts at the top of the frame."""
        alerts = []

        for e in nm_events[:2]:   # max 2 near-miss alerts shown
            if e.alert_level == AlertLevel.CRITICAL:
                alerts.append((e.description, COLOR["box_critical"]))
            elif e.alert_level == AlertLevel.HIGH:
                alerts.append((e.description, COLOR["box_high"]))
            else:
                alerts.append((e.description, COLOR["warning"]))

        for e in beh_events[:2]:  # max 2 behavior alerts shown
            if e.behavior == BehaviorType.SUDDEN_BRAKING:
                alerts.append((e.description, COLOR["braking"]))
            elif e.behavior == BehaviorType.WRONG_WAY:
                alerts.append((e.description, COLOR["wrong_way"]))
            elif e.behavior == BehaviorType.STOPPED_VEHICLE:
                alerts.append((e.description, COLOR["stopped"]))
            elif e.behavior == BehaviorType.TAILGATING:
                alerts.append((e.description, COLOR["tailgate"]))
            else:
                alerts.append((e.description, COLOR["warning"]))

        for i, (text, color) in enumerate(alerts[:3]):
            y = 22 + i * 24
            cv2.rectangle(frame, (0, y-16), (frame.shape[1], y+6),
                         (0,0,0), -1)
            cv2.putText(
                frame, text[:80],
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1, cv2.LINE_AA
            )

    def _draw_stats(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        frame_idx: int
    ):
        """Draw frame statistics at bottom left."""
        h = frame.shape[0]
        stats = [
            f"Frame: {frame_idx}",
            f"Vehicles: {len(tracks)}",
            f"Incidents: {self.total_incidents}",
        ]
        for i, s in enumerate(stats):
            y = h - 12 - (len(stats) - 1 - i) * 18
            cv2.putText(
                frame, s, (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                COLOR["id_text"], 1, cv2.LINE_AA
            )

    def _assemble_video(self, sequence_id: str):
        """Assemble annotated frames into mp4 video using OpenCV."""
        frame_paths = sorted(
            glob.glob(str(self.output_dir / "frame_*.jpg"))
        )
        if not frame_paths:
            print("No frames to assemble")
            return

        # Read first frame to get dimensions
        sample = cv2.imread(frame_paths[0])
        h, w = sample.shape[:2]

        video_path = str(self.output_dir / f"{sequence_id}_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, self.fps, (w, h))

        print(f"\nAssembling video: {video_path}")
        for path in frame_paths:
            img = cv2.imread(path)
            if img is not None:
                writer.write(img)
        writer.release()

        size_mb = os.path.getsize(video_path) / 1e6
        print(f"Video saved: {size_mb:.1f} MB ✅")

    def _print_summary(self, sequence_id: str):
        """Print final processing summary."""
        nm_summary  = self.near_miss.get_summary()
        beh_summary = self.classifier.get_summary()

        print(f"\n{'='*50}")
        print(f"PIPELINE SUMMARY — {sequence_id}")
        print(f"{'='*50}")
        print(f"Frames processed:  {self.total_frames}")
        print(f"Total incidents:   {self.total_incidents}")
        print(f"\nNear-miss events:")
        for k, v in nm_summary.items():
            print(f"  {k}: {v}")
        print(f"\nBehavior events:")
        for k, v in beh_summary.items():
            print(f"  {k}: {v}")
        print(f"{'='*50}\n")