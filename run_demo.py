import os
import sys
sys.path.insert(0, '.')

from detection.pipeline import HighwaySafetyPipeline
from storage.event_publisher import IncidentPublisher
from storage.bigquery_logger import BigQueryLogger

SEQUENCE_DIR = (
    "data/raw/ua-detrac/content/UA-DETRAC/"
    "DETRAC_Upload/images/train"
)

SEQUENCES = [
    ("MVI_20011", 150),   # dense traffic — most incidents
    ("MVI_20032", 150),   # different location
    ("MVI_39781", 150),   # varied conditions
]

os.makedirs("output/demo", exist_ok=True)

for seq_id, max_frames in SEQUENCES:
    print(f"\n{'='*50}")
    print(f"Processing {seq_id} ({max_frames} frames)")
    print('='*50)

    pipeline = HighwaySafetyPipeline(
        model_path="models/best_yolov8s_v1.pt",
        fps=25.0,
        pixels_per_meter=8.0,
        confidence_threshold=0.4,
        output_dir=f"output/demo/{seq_id}"
    )

    results = pipeline.run_sequence(
        sequence_dir=SEQUENCE_DIR,
        sequence_id=seq_id,
        max_frames=max_frames,
        save_video=True
    )

    nm  = pipeline.near_miss.get_summary()
    beh = pipeline.classifier.get_summary()

    print(f"\nResults for {seq_id}:")
    print(f"  Frames:    {len(results)}")
    print(f"  Incidents: {nm.get('total_events',0)} near-miss, "
          f"{beh.get('total_behaviors',0)} behavior")
    print(f"  Min TTC:   {nm.get('min_ttc','N/A')}")

print("\nAll sequences done.")
print("Videos saved in output/demo/")