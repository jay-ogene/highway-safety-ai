from detection.pipeline import HighwaySafetyPipeline
from storage.event_publisher import IncidentPublisher
from storage.bigquery_logger import BigQueryLogger
from storage.model_uploader import ModelUploader

# Upload model to GCS first (do this once)
uploader = ModelUploader()
uploader.upload_model(
    local_path="models/best_yolov8s_v1.pt",
    model_name="best_yolov8s_v1.pt",
    version_tag="v1"
)

# Initialize cloud loggers
publisher = IncidentPublisher()
bq_logger  = BigQueryLogger()

# Run pipeline
pipeline = HighwaySafetyPipeline(
    model_path="models/best_yolov8s_v1.pt",
    fps=25.0,
    pixels_per_meter=8.0,
    confidence_threshold=0.4,
    output_dir="output/pipeline"
)

SEQUENCE = "MVI_20011"
results = pipeline.run_sequence(
    sequence_dir="data/raw/ua-detrac/content/UA-DETRAC/"
                 "DETRAC_Upload/images/train",
    sequence_id=SEQUENCE,
    max_frames=100,
    save_video=True
)

# Stream all events to cloud
print("\nStreaming events to GCP...")
nm_published = 0
beh_published = 0

for result in results:
    # Only publish CRITICAL and HIGH near-misses to avoid noise
    for event in result.near_miss_events:
        if event.alert_level.value in ["CRITICAL", "HIGH"]:
            publisher.publish_near_miss(event, SEQUENCE)
            bq_logger.log_near_miss(event, SEQUENCE)
            nm_published += 1

    # Publish all behavior events
    for event in result.behavior_events:
        publisher.publish_behavior(event, SEQUENCE)
        bq_logger.log_behavior(event, SEQUENCE)
        beh_published += 1

    # Frame summary every 10 frames
    if result.frame_idx % 10 == 0:
        publisher.publish_frame_summary(
            frame_idx=result.frame_idx,
            vehicle_count=len(result.tracks),
            incident_count=len(result.near_miss_events) +
                           len(result.behavior_events),
            sequence_id=SEQUENCE
        )

# Flush remaining BigQuery rows
bq_logger.flush()

print(f"\nCloud upload complete:")
print(f"  Near-miss events published: {nm_published}")
print(f"  Behavior events published:  {beh_published}")
print(f"  BigQuery rows logged:       {bq_logger.total_logged}")
print(f"\nPub/Sub stats: {publisher.get_stats()}")