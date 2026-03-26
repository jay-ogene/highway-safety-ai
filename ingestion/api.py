import os
import sys
from datetime import datetime
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from storage.bigquery_logger import BigQueryLogger
from ingestion.schemas import (
    AnalyzeRequest, AnalyzeResponse,
    IncidentSummary, IncidentRecord,
    HealthResponse
)

app = FastAPI(
    title="Highway Safety AI API",
    description=(
        "Real-time highway incident detection and analysis. "
        "Detects near-misses, tailgating, sudden braking, "
        "wrong-way driving, and stopped vehicles using "
        "YOLOv8s trained on 140,000 highway frames."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load heavy components
_pipeline = None
_bq_logger = None


def get_bq_logger() -> BigQueryLogger:
    global _bq_logger
    if _bq_logger is None:
        _bq_logger = BigQueryLogger()
    return _bq_logger


def get_pipeline(model_path: str = "models/best_yolov8s_v1.pt"):
    global _pipeline
    if _pipeline is None:
        from detection.pipeline import HighwaySafetyPipeline
        _pipeline = HighwaySafetyPipeline(
            model_path=model_path,
            fps=25.0,
            pixels_per_meter=8.0,
            confidence_threshold=0.4,
            output_dir="output/api"
        )
    return _pipeline


# ── Routes ────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """
    Check API health and GCP connectivity.
    Use this to verify the service is running correctly.
    """
    try:
        bq = get_bq_logger()
        gcp_status = "connected"
    except Exception as e:
        gcp_status = f"error: {str(e)[:80]}"

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model="YOLOv8s — 0.833 mAP@0.5",
        gcp_project=os.getenv("GCP_PROJECT_ID", "highway-safety-ai-jude"),
        gcp_status=gcp_status,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


@app.get(
    "/incidents",
    response_model=List[IncidentRecord],
    tags=["Incidents"]
)
async def get_incidents(
    limit: int = Query(20, ge=1, le=500,
                       description="Number of records to return"),
    severity: Optional[int] = Query(None, ge=1, le=4,
                       description="Filter by severity 1-4"),
    event_type: Optional[str] = Query(None,
                       description="Filter: near_miss or behavior"),
    sequence_id: Optional[str] = Query(None,
                       description="Filter by camera sequence")
):
    """
    Retrieve recent incidents from BigQuery.

    Severity levels:
    - 1 = Safe
    - 2 = Warning
    - 3 = High
    - 4 = Critical
    """
    try:
        bq = get_bq_logger()

        # Build filtered query
        conditions = []
        if severity:
            conditions.append(f"severity = {severity}")
        if event_type:
            conditions.append(f"event_type = '{event_type}'")
        if sequence_id:
            conditions.append(f"sequence_id = '{sequence_id}'")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        sql = f"""
            SELECT
                event_id, event_type, alert_level, behavior_type,
                severity, ttc_seconds, distance_meters,
                closing_speed_kmh, vehicle_a_id, vehicle_b_id,
                track_id, frame_idx, sequence_id, camera_id,
                description, timestamp
            FROM `{bq.table_ref}`
            {where}
            ORDER BY timestamp DESC
            LIMIT {limit}
        """
        rows = list(bq.client.query(sql).result())
        return [IncidentRecord(**dict(r)) for r in rows]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/incidents/summary",
    response_model=IncidentSummary,
    tags=["Incidents"]
)
async def get_summary(
    sequence_id: Optional[str] = Query(None,
                        description="Filter by camera sequence")
):
    """
    Aggregated incident statistics from BigQuery.
    Returns counts by type, severity distribution,
    and average TTC for near-miss events.
    """
    try:
        bq = get_bq_logger()
        seq_filter = (
            f"WHERE sequence_id = '{sequence_id}'"
            if sequence_id else ""
        )

        sql = f"""
            SELECT
                COUNT(*) as total_incidents,
                COUNTIF(event_type = 'near_miss') as near_miss_count,
                COUNTIF(event_type = 'behavior') as behavior_count,
                COUNTIF(severity = 4) as critical_count,
                COUNTIF(severity = 3) as high_count,
                COUNTIF(severity = 2) as warning_count,
                COUNTIF(behavior_type = 'TAILGATING') as tailgating_count,
                COUNTIF(behavior_type = 'SUDDEN_BRAKING') as braking_count,
                COUNTIF(behavior_type = 'WRONG_WAY') as wrong_way_count,
                COUNTIF(behavior_type = 'STOPPED_VEHICLE') as stopped_count,
                ROUND(AVG(
                    IF(ttc_seconds > 0, ttc_seconds, NULL)
                ), 3) as avg_ttc,
                ROUND(MIN(
                    IF(ttc_seconds > 0, ttc_seconds, NULL)
                ), 3) as min_ttc,
                COUNT(DISTINCT sequence_id) as sequences_analyzed,
                COUNT(DISTINCT camera_id) as cameras_active
            FROM `{bq.table_ref}`
            {seq_filter}
        """
        rows = list(bq.client.query(sql).result())
        row = dict(rows[0]) if rows else {}

        return IncidentSummary(
            total_incidents=row.get("total_incidents", 0),
            near_miss_count=row.get("near_miss_count", 0),
            behavior_count=row.get("behavior_count", 0),
            critical_count=row.get("critical_count", 0),
            high_count=row.get("high_count", 0),
            warning_count=row.get("warning_count", 0),
            tailgating_count=row.get("tailgating_count", 0),
            sudden_braking_count=row.get("braking_count", 0),
            wrong_way_count=row.get("wrong_way_count", 0),
            stopped_vehicle_count=row.get("stopped_count", 0),
            avg_ttc_seconds=row.get("avg_ttc"),
            min_ttc_seconds=row.get("min_ttc"),
            sequences_analyzed=row.get("sequences_analyzed", 0),
            cameras_active=row.get("cameras_active", 0)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    tags=["Analysis"]
)
async def analyze_sequence(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger pipeline analysis on a UA-DETRAC sequence.

    Runs detection, tracking, and incident classification
    on the specified sequence, streams results to GCP,
    and returns a summary.
    """
    sequence_dir = (
        "data/raw/ua-detrac/content/UA-DETRAC/"
        "DETRAC_Upload/images/train"
    )

    # Check sequence exists
    import glob
    pattern = f"{sequence_dir}/{request.sequence_id}_img*.jpg"
    frames = glob.glob(pattern)
    if not frames:
        raise HTTPException(
            status_code=404,
            detail=f"Sequence {request.sequence_id} not found"
        )

    try:
        pipeline = get_pipeline(request.model_path)
        from storage.event_publisher import IncidentPublisher
        publisher = IncidentPublisher()
        bq = get_bq_logger()

        results = pipeline.run_sequence(
            sequence_dir=sequence_dir,
            sequence_id=request.sequence_id,
            max_frames=request.max_frames,
            save_video=request.save_video
        )

        # Stream to cloud
        nm_count = beh_count = 0
        for result in results:
            for event in result.near_miss_events:
                if event.alert_level.value in ["CRITICAL", "HIGH"]:
                    publisher.publish_near_miss(
                        event, request.sequence_id
                    )
                    bq.log_near_miss(event, request.sequence_id)
                    nm_count += 1
            for event in result.behavior_events:
                publisher.publish_behavior(
                    event, request.sequence_id
                )
                bq.log_behavior(event, request.sequence_id)
                beh_count += 1

        bq.flush()

        nm_summary = pipeline.near_miss.get_summary()
        beh_summary = pipeline.classifier.get_summary()

        return AnalyzeResponse(
            sequence_id=request.sequence_id,
            frames_processed=len(results),
            total_incidents=nm_count + beh_count,
            near_miss_events=nm_count,
            behavior_events=beh_count,
            min_ttc=nm_summary.get("min_ttc"),
            avg_ttc=nm_summary.get("avg_ttc"),
            pairs_involved=nm_summary.get("pairs_involved", 0),
            behaviors_detected=beh_summary.get("by_type", {}),
            status="complete",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))