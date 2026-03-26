from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    gcp_project: str
    gcp_status: str
    timestamp: str


class IncidentRecord(BaseModel):
    event_id: str
    event_type: str
    alert_level: str
    behavior_type: Optional[str]
    severity: int
    ttc_seconds: Optional[float]
    distance_meters: Optional[float]
    closing_speed_kmh: Optional[float]
    vehicle_a_id: Optional[int]
    vehicle_b_id: Optional[int]
    track_id: Optional[int]
    frame_idx: int
    sequence_id: str
    camera_id: str
    description: str
    timestamp: Optional[datetime]


class IncidentSummary(BaseModel):
    total_incidents: int
    near_miss_count: int
    behavior_count: int
    critical_count: int
    high_count: int
    warning_count: int
    tailgating_count: int
    sudden_braking_count: int
    wrong_way_count: int
    stopped_vehicle_count: int
    avg_ttc_seconds: Optional[float]
    min_ttc_seconds: Optional[float]
    sequences_analyzed: int
    cameras_active: int


class AnalyzeRequest(BaseModel):
    sequence_id: str = Field(
        default="MVI_20011",
        description="UA-DETRAC sequence ID e.g. MVI_20011"
    )
    max_frames: Optional[int] = Field(
        default=100,
        description="Max frames to process (None = all)"
    )
    save_video: bool = Field(
        default=True,
        description="Save annotated output video"
    )
    model_path: str = Field(
        default="models/best_yolov8s_v1.pt",
        description="Path to YOLO weights file"
    )


class AnalyzeResponse(BaseModel):
    sequence_id: str
    frames_processed: int
    total_incidents: int
    near_miss_events: int
    behavior_events: int
    min_ttc: Optional[float]
    avg_ttc: Optional[float]
    pairs_involved: int
    behaviors_detected: Dict[str, int]
    status: str
    timestamp: str