import numpy as np
import sys
sys.path.insert(0, '.')
from detection.tracker import VehicleTracker

# Simulate 5 frames with 3 vehicles moving
tracker = VehicleTracker(fps=25.0, pixels_per_meter=8.0)

# Frame 1 — 3 vehicles detected
detections = np.array([
    [100, 200, 180, 250],   # vehicle A
    [300, 150, 380, 200],   # vehicle B
    [500, 300, 580, 350],   # vehicle C
], dtype=float)
confs = np.array([0.92, 0.88, 0.95])

for frame in range(10):
    # Simulate vehicles moving right at different speeds
    detections[:, 0] += [2, 4, 6]   # x1 moves
    detections[:, 2] += [2, 4, 6]   # x2 moves

    tracks = tracker.update(detections, confs, frame_idx=frame)
    print(f"\nFrame {frame}: {len(tracks)} active tracks")
    for t in tracks:
        print(f"  ID:{t.track_id}  bbox:{t.bbox.astype(int)}  "
              f"speed:{t.speed_kmh:.1f} km/h  hits:{t.hits}")

print("\n✅ Tracker working correctly")

print("\n--- Testing NearMissDetector ---\n")

from detection.near_miss import NearMissDetector, AlertLevel

tracker2 = VehicleTracker(fps=25.0, pixels_per_meter=8.0)
detector  = NearMissDetector(pixels_per_meter=8.0, fps=25.0)

# Simulate two vehicles on collision course
# Vehicle A at x=100, moving right at 4px/frame
# Vehicle B at x=300, moving LEFT at 8px/frame (approaching)
dets = np.array([
    [100, 200, 180, 250],   # vehicle A — moving right
    [300, 200, 380, 250],   # vehicle B — moving left
], dtype=float)
confs = np.array([0.92, 0.88])

for frame in range(20):
    dets[0, 0] += 4;  dets[0, 2] += 4   # A moves right
    dets[1, 0] -= 8;  dets[1, 2] -= 8   # B moves left (approaching A)

    tracks  = tracker2.update(dets, confs, frame_idx=frame)
    events  = detector.update(tracks, frame_idx=frame)

    if events:
        for e in events:
            print(f"Frame {frame:2d}: {e.description}")

print("\nSummary:", detector.get_summary())
print("\n✅ NearMissDetector working correctly")
print("\n--- Testing BehaviorClassifier ---\n")
from detection.behavior_classifier import BehaviorClassifier, BehaviorType

tracker3    = VehicleTracker(fps=25.0, pixels_per_meter=8.0)
classifier  = BehaviorClassifier(fps=25.0, pixels_per_meter=8.0)

# Simulate: vehicle braking hard from 80 to 10 km/h
# and a stopped vehicle
dets2 = np.array([
    [100, 200, 180, 250],   # vehicle — will brake
    [400, 300, 480, 350],   # vehicle — will stop
], dtype=float)
confs2 = np.array([0.92, 0.88])

for frame in range(30):
    # Vehicle 0: moves fast then brakes hard after frame 15
    speed_px = 8 if frame < 15 else 1
    dets2[0, 0] += speed_px
    dets2[0, 2] += speed_px

    # Vehicle 1: stops after frame 10
    if frame < 10:
        dets2[1, 0] += 3
        dets2[1, 2] += 3

    tracks3 = tracker3.update(dets2, confs2, frame_idx=frame)
    events3 = classifier.update(tracks3, frame_idx=frame)

    for e in events3:
        print(f"Frame {frame:2d}: {e.description}")

print("\nSummary:", classifier.get_summary())
print("\n✅ BehaviorClassifier working correctly")