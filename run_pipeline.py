from detection.pipeline import HighwaySafetyPipeline

pipeline = HighwaySafetyPipeline(
    model_path="models/best_yolov8s_v1.pt",
    fps=25.0,
    pixels_per_meter=8.0,
    confidence_threshold=0.4,
    output_dir="output/pipeline"
)

pipeline.run_sequence(
    sequence_dir="data/raw/ua-detrac/content/UA-DETRAC/"
                 "DETRAC_Upload/images/train",   # ← train not val
    sequence_id="MVI_20011",
    max_frames=100,
    save_video=True
)