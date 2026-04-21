import cv2
import os
import numpy as np

OUTPUT_DIR = "output/demo"
FINAL_VIDEO = "output/demo/highway_safety_demo.mp4"
FPS = 25

def text_card(lines, frames_count, width, height):
    """Black card with multiple lines of text."""
    frames = []
    img = np.zeros((height, width, 3), dtype=np.uint8)
    y = height // 2 - (len(lines) * 52) // 2
    for text, scale, color in lines:
        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            thick = 2 if scale >= 0.8 else 1
            tw = cv2.getTextSize(text, font, scale, thick)[0][0]
            x = (width - tw) // 2
            cv2.putText(img, text, (x, y), font,
                       scale, color, thick, cv2.LINE_AA)
        y += 52
    for _ in range(frames_count):
        frames.append(img.copy())
    return frames


def load_frames(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def banner(frame, text, color=(100, 220, 100)):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(frame.copy(), 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, text, (12, 27),
               cv2.FONT_HERSHEY_SIMPLEX, 0.62,
               color, 1, cv2.LINE_AA)
    return frame


def legend_overlay(frame, items):
    """Draw a legend box bottom-left."""
    h, w = frame.shape[:2]
    box_h = len(items) * 22 + 16
    cv2.rectangle(frame, (8, h-box_h-8), (370, h-8),
                 (0, 0, 0), -1)
    cv2.rectangle(frame, (8, h-box_h-8), (370, h-8),
                 (80, 80, 80), 1)
    for i, (symbol, text, color) in enumerate(items):
        y = h - box_h + 6 + i * 22
        cv2.putText(frame, symbol, (16, y+14),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                   color, 1, cv2.LINE_AA)
        cv2.putText(frame, text, (50, y+14),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                   (200, 200, 200), 1, cv2.LINE_AA)
    return frame


def stats_box(frame, title, stats):
    """Draw stats panel top-right."""
    h, w = frame.shape[:2]
    box_h = len(stats) * 22 + 34
    box_w = 260
    x0 = w - box_w - 8
    cv2.rectangle(frame, (x0, 48), (w-8, 48+box_h),
                 (0, 0, 0), -1)
    cv2.rectangle(frame, (x0, 48), (w-8, 48+box_h),
                 (80, 80, 80), 1)
    cv2.putText(frame, title, (x0+8, 66),
               cv2.FONT_HERSHEY_SIMPLEX, 0.48,
               (180, 180, 180), 1, cv2.LINE_AA)
    for i, (label, val, color) in enumerate(stats):
        y = 88 + i * 22
        cv2.putText(frame, f"{label}: {val}",
                   (x0+8, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                   color, 1, cv2.LINE_AA)
    return frame


# Get dimensions
sample_cap = cv2.VideoCapture(
    f"{OUTPUT_DIR}/MVI_20011/MVI_20011_output.mp4"
)
ret, sf = sample_cap.read()
H, W = sf.shape[:2]
sample_cap.release()

print(f"Frame size: {W}x{H}")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(FINAL_VIDEO, fourcc, FPS, (W, H))

WHITE  = (255, 255, 255)
GRAY   = (180, 180, 180)
DIM    = (120, 120, 120)
TEAL   = (180, 220, 100)
RED    = (80,  80,  255)
ORANGE = (80, 165, 255)
GREEN  = (100, 220, 100)
BLUE   = (255, 160, 80)

total_frames = 0

def write(frames):
    global total_frames
    for f in frames:
        writer.write(f)
    total_frames += len(frames)

# ─────────────────────────────────────────────────────────
# OPENING TITLE  ~5 seconds
# ─────────────────────────────────────────────────────────
write(text_card([
    ("AI Highway Safety Platform", 1.1,  WHITE),
    ("", 0, WHITE),
    ("Real-time vehicle detection · tracking · incident alerts", 0.56, GRAY),
    ("", 0, WHITE),
    ("YOLOv8s  ·  0.833 mAP@0.5  ·  Google Cloud", 0.58, TEAL),
], 125, W, H))

# ─────────────────────────────────────────────────────────
# PROBLEM STATEMENT  ~6 seconds
# ─────────────────────────────────────────────────────────
write(text_card([
    ("38,000 people die on US roads every year.", 0.78, WHITE),
    ("", 0, WHITE),
    ("Many crashes are preceded by detectable warning signs:", 0.56, GRAY),
    ("tailgating · sudden braking · dangerous following distance", 0.54, GRAY),
    ("", 0, WHITE),
    ("This system detects those signs — before the crash.", 0.58, TEAL),
], 150, W, H))

# ─────────────────────────────────────────────────────────
# HOW IT WORKS  ~5 seconds
# ─────────────────────────────────────────────────────────
write(text_card([
    ("How it works", 0.9, WHITE),
    ("", 0, WHITE),
    ("Step 1: YOLOv8s detects every vehicle in each frame", 0.54, GRAY),
    ("Step 2: Kalman filter tracks vehicles across frames with persistent IDs", 0.50, GRAY),
    ("Step 3: Time-To-Collision calculated for every vehicle pair", 0.50, GRAY),
    ("Step 4: Dangerous behaviour classified and streamed to GCP", 0.50, GRAY),
], 125, W, H))

# ─────────────────────────────────────────────────────────
# READING THE OVERLAY  ~5 seconds
# ─────────────────────────────────────────────────────────
write(text_card([
    ("Reading the overlay", 0.9, WHITE),
    ("", 0, WHITE),
    ("ID:3  52km/h        — vehicle identity + estimated speed", 0.54, GRAY),
    ("White box           — normal, safe following distance", 0.54, GREEN),
    ("Orange box          — HIGH RISK  (TTC < 2.5 seconds)", 0.54, ORANGE),
    ("Red box             — NEAR MISS  (TTC < 1.5 seconds)", 0.54, RED),
    ("Green alert         — tailgating detected", 0.54, GREEN),
], 125, W, H))

# ─────────────────────────────────────────────────────────
# SEQUENCE 1: Light traffic  ~8 seconds of footage
# ─────────────────────────────────────────────────────────
write(text_card([
    ("Sequence 1 — Light traffic", 0.9, WHITE),
    ("Camera: MVI_20032", 0.55, GRAY),
    ("", 0, WHITE),
    ("Watch how each vehicle gets a persistent ID", 0.56, GRAY),
    ("and a real-time speed estimate in km/h.", 0.56, GRAY),
], 75, W, H))

frames_light = load_frames(
    f"{OUTPUT_DIR}/MVI_20032/MVI_20032_output.mp4", 200
)
legend = [
    ("ID:N", "Persistent vehicle identity (Kalman filter)", GREEN),
    ("km/h", "Speed estimated from pixel displacement", WHITE),
    ("box ", "Color = alert level (white=safe, red=critical)", ORANGE),
]
for f in frames_light:
    f = banner(f, "VEHICLE TRACKING  |  Persistent IDs + Speed Estimation",
               color=GREEN)
    f = legend_overlay(f, legend)
    writer.write(f)
    total_frames += 1

# ─────────────────────────────────────────────────────────
# SEQUENCE 1 SUMMARY
# ─────────────────────────────────────────────────────────
write(text_card([
    ("Sequence 1 result", 0.8, WHITE),
    ("", 0, WHITE),
    ("5 vehicles tracked simultaneously with persistent IDs", 0.54, GRAY),
    ("Speed estimates: 20-55 km/h  (consistent with congested highway)", 0.52, GRAY),
    ("4 stopped-vehicle alerts detected", 0.52, TEAL),
], 75, W, H))

# ─────────────────────────────────────────────────────────
# SEQUENCE 2: Dense traffic  ~10 seconds
# ─────────────────────────────────────────────────────────
write(text_card([
    ("Sequence 2 — Dense traffic", 0.9, WHITE),
    ("Camera: MVI_20011", 0.55, GRAY),
    ("", 0, WHITE),
    ("101 vehicle pairs monitored simultaneously.", 0.56, GRAY),
    ("Watch TTC alerts fire when vehicles get dangerously close.", 0.54, GRAY),
    ("Min TTC recorded: 0.13 seconds.", 0.56, RED),
], 100, W, H))

frames_dense = load_frames(
    f"{OUTPUT_DIR}/MVI_20011/MVI_20011_output.mp4", 250
)
for f in frames_dense:
    f = banner(f,
               "NEAR-MISS DETECTION  |  101 pairs  |  min TTC: 0.13s",
               color=RED)
    f = stats_box(f, "Live alert summary", [
        ("CRITICAL (TTC<1.5s)", "297",  RED),
        ("HIGH     (TTC<2.5s)", "1848", ORANGE),
        ("WARNING  (TTC<3.0s)", "814",  BLUE),
        ("Pairs monitored",     "101",  GRAY),
    ])
    writer.write(f)
    total_frames += 1

# ─────────────────────────────────────────────────────────
# SEQUENCE 2 SUMMARY
# ─────────────────────────────────────────────────────────
write(text_card([
    ("Sequence 2 result", 0.8, WHITE),
    ("", 0, WHITE),
    ("297 CRITICAL near-miss alerts in 150 frames", 0.56, RED),
    ("Minimum TTC: 0.13 seconds — 130ms from collision", 0.54, GRAY),
    ("Average TTC: 2.13 seconds across all pairs", 0.54, GRAY),
    ("", 0, WHITE),
    ("Every alert streamed to GCP Pub/Sub in real time", 0.52, TEAL),
], 100, W, H))

# ─────────────────────────────────────────────────────────
# SEQUENCE 3: Behaviour detection  ~10 seconds
# ─────────────────────────────────────────────────────────
write(text_card([
    ("Sequence 3 — Behaviour classification", 0.9, WHITE),
    ("Camera: MVI_39781", 0.55, GRAY),
    ("", 0, WHITE),
    ("Beyond near-misses — classifying dangerous driving patterns.", 0.54, GRAY),
    ("Watch for: unsafe lane changes · sudden braking · wrong-way", 0.52, ORANGE),
], 100, W, H))

frames_beh = load_frames(
    f"{OUTPUT_DIR}/MVI_39781/MVI_39781_output.mp4", 250
)
beh_legend = [
    ("LANE", "Unsafe lane change (rapid lateral movement at speed)", ORANGE),
    ("BRAKE", "Sudden braking (decel > 15 km/h per second)", RED),
    ("WRONG", "Wrong-way vehicle (opposing dominant traffic flow)", RED),
]
for f in frames_beh:
    f = banner(f,
               "BEHAVIOUR DETECTION  |  lane change · braking · wrong-way",
               color=ORANGE)
    f = legend_overlay(f, beh_legend)
    f = stats_box(f, "Behaviours detected", [
        ("Unsafe lane change", "8",  ORANGE),
        ("Sudden braking",     "49", RED),
        ("Wrong-way driving",  "2",  RED),
    ])
    writer.write(f)
    total_frames += 1

# ─────────────────────────────────────────────────────────
# SEQUENCE 3 SUMMARY
# ─────────────────────────────────────────────────────────
write(text_card([
    ("Sequence 3 result", 0.8, WHITE),
    ("", 0, WHITE),
    ("8 unsafe lane changes detected", 0.54, ORANGE),
    ("49 sudden braking events flagged", 0.54, RED),
    ("2 wrong-way vehicles identified", 0.54, RED),
    ("", 0, WHITE),
    ("All events logged to BigQuery · streamed via Pub/Sub", 0.52, TEAL),
], 100, W, H))

# ─────────────────────────────────────────────────────────
# NGSIM VALIDATION  ~5 seconds
# ─────────────────────────────────────────────────────────
write(text_card([
    ("Ground truth validation", 0.88, WHITE),
    ("", 0, WHITE),
    ("Alert thresholds validated against NGSIM I-80 dataset", 0.56, GRAY),
    ("8,305,244 real vehicle observations  —  US Dept. of Transportation", 0.52, GRAY),
    ("", 0, WHITE),
    ("11.5% of real I-80 vehicles operate at CRITICAL TTC levels", 0.54, RED),
    ("Tailgating threshold (10m) flags 9.7% of real following pairs", 0.52, TEAL),
], 125, W, H))

# ─────────────────────────────────────────────────────────
# SYSTEM OVERVIEW  ~5 seconds
# ─────────────────────────────────────────────────────────
write(text_card([
    ("Full production stack", 0.88, WHITE),
    ("", 0, WHITE),
    ("YOLOv8s  ·  ByteTrack  ·  Kalman filter  ·  FastAPI  ·  Docker", 0.52, GRAY),
    ("GCP: Cloud Run  ·  Pub/Sub  ·  BigQuery  ·  Artifact Registry", 0.52, TEAL),
    ("Streamlit dashboard  ·  REST API  ·  1,553 incidents logged", 0.52, GRAY),
    ("", 0, WHITE),
    ("Trained on 140,000 frames  ·  0.833 mAP@0.5", 0.58, WHITE),
], 125, W, H))

# ─────────────────────────────────────────────────────────
# END CARD  ~6 seconds
# ─────────────────────────────────────────────────────────
write(text_card([
    ("Jude Ogene", 1.0,  WHITE),
    ("AI Engineer  ·  Atlanta, GA", 0.62, GRAY),
    ("M.S. Computer Science — Kennesaw State University", 0.50, DIM),
    ("", 0, WHITE),
    ("github.com/jay-ogene/highway-safety-ai", 0.58, TEAL),
    ("", 0, WHITE),
    ("Live API: highway-safety-api-720304622514.us-central1.run.app/docs", 0.42, GRAY),
    ("", 0, WHITE),
    ("Open to AI Engineer roles  ·  Requires H-1B sponsorship", 0.50, TEAL),
], 150, W, H))

writer.release()

duration = total_frames / FPS
size_mb = os.path.getsize(FINAL_VIDEO) / 1e6
print(f"\nDemo video complete!")
print(f"File:     {FINAL_VIDEO}")
print(f"Size:     {size_mb:.1f} MB")
print(f"Duration: {duration:.0f} seconds ({duration/60:.1f} minutes)")
print(f"Frames:   {total_frames}")