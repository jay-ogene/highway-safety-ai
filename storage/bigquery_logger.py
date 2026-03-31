import os
import json
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


class BigQueryLogger:
    """
    Logs highway safety incidents to BigQuery.

    Every incident becomes one row in the incidents table.
    Rows are batched and inserted together for efficiency.
    You can then query the table with standard SQL.
    """

    # BigQuery schema for the incidents table
    SCHEMA = [
        {"name": "event_id",          "type": "STRING"},
        {"name": "event_type",         "type": "STRING"},
        {"name": "timestamp",          "type": "TIMESTAMP"},
        {"name": "sequence_id",        "type": "STRING"},
        {"name": "camera_id",          "type": "STRING"},
        {"name": "frame_idx",          "type": "INTEGER"},
        {"name": "vehicle_a_id",       "type": "INTEGER"},
        {"name": "vehicle_b_id",       "type": "INTEGER"},
        {"name": "track_id",           "type": "INTEGER"},
        {"name": "ttc_seconds",        "type": "FLOAT"},
        {"name": "distance_meters",    "type": "FLOAT"},
        {"name": "closing_speed_kmh",  "type": "FLOAT"},
        {"name": "alert_level",        "type": "STRING"},
        {"name": "behavior_type",      "type": "STRING"},
        {"name": "confidence",         "type": "FLOAT"},
        {"name": "severity",           "type": "INTEGER"},
        {"name": "description",        "type": "STRING"},
    ]

    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        table_id: Optional[str] = None,
        credentials_path: Optional[str] = None
    ):
        self.project_id = project_id or os.getenv(
            'GCP_PROJECT_ID', 'highway-safety-ai-jude'
        )
        self.dataset_id = dataset_id or os.getenv(
            'BQ_DATASET', 'highway_safety'
        )
        self.table_id = table_id or os.getenv(
            'BQ_EVENTS_TABLE', 'incidents'
        )
        # credentials_path = credentials_path or os.getenv(
        #     'GOOGLE_APPLICATION_CREDENTIALS', './secrets/gcp-key.json'
        # )

        # if credentials_path and os.path.exists(credentials_path):
        #     os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        credentials_path = credentials_path or os.getenv(
             'GOOGLE_APPLICATION_CREDENTIALS', ''
        )

        if credentials_path and os.path.exists(credentials_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        # else: use Application Default Credentials (Cloud Run uses these automatically)

        from google.cloud import bigquery
        self.client = bigquery.Client(project=self.project_id)
        self.table_ref = (
            f"{self.project_id}.{self.dataset_id}.{self.table_id}"
        )

        # Batch buffer
        self.batch: List[dict] = []
        self.batch_size = 100       # insert every 100 rows
        self.total_logged = 0

        # Ensure dataset + table exist
        self._ensure_table()
        print(f"BigQueryLogger ready → {self.table_ref}")

    def _ensure_table(self):
        """Create BigQuery dataset and table if they don't exist."""
        from google.cloud import bigquery
        from google.api_core.exceptions import NotFound

        # Dataset
        dataset_ref = self.client.dataset(self.dataset_id)
        try:
            self.client.get_dataset(dataset_ref)
            print(f"Dataset exists: {self.dataset_id} ✅")
        except NotFound:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            self.client.create_dataset(dataset)
            print(f"Created dataset: {self.dataset_id}")

        # Table
        table_ref = self.client.dataset(
            self.dataset_id
        ).table(self.table_id)
        try:
            self.client.get_table(table_ref)
            print(f"Table exists: {self.table_id} ✅")
        except NotFound:
            from google.cloud.bigquery import SchemaField
            schema = [
                SchemaField(f["name"], f["type"])
                for f in self.SCHEMA
            ]
            table = bigquery.Table(table_ref, schema=schema)
            self.client.create_table(table)
            print(f"Created table: {self.table_id}")

    def log_near_miss(
        self,
        event,
        sequence_id: str,
        camera_id: str = "UA-DETRAC-CAM"
    ):
        """Add a near-miss event to the batch buffer."""
        import uuid
        row = {
            "event_id":         str(uuid.uuid4()),
            "event_type":       "near_miss",
            "timestamp":        datetime.utcnow().isoformat(),
            "sequence_id":      sequence_id,
            "camera_id":        camera_id,
            "frame_idx":        event.frame_idx,
            "vehicle_a_id":     event.vehicle_a_id,
            "vehicle_b_id":     event.vehicle_b_id,
            "track_id":         -1,
            "ttc_seconds":      round(event.ttc_seconds, 3),
            "distance_meters":  round(event.distance_meters, 2),
            "closing_speed_kmh": round(event.closing_speed_kmh, 1),
            "alert_level":      event.alert_level.value,
            "behavior_type":    "near_miss",
            "confidence":       1.0,
            "severity":         self._alert_to_severity(
                                    event.alert_level.value
                                ),
            "description":      event.description
        }
        self._add_to_batch(row)

    def log_behavior(
        self,
        event,
        sequence_id: str,
        camera_id: str = "UA-DETRAC-CAM"
    ):
        """Add a behavior event to the batch buffer."""
        import uuid
        row = {
            "event_id":         str(uuid.uuid4()),
            "event_type":       "behavior",
            "timestamp":        datetime.utcnow().isoformat(),
            "sequence_id":      sequence_id,
            "camera_id":        camera_id,
            "frame_idx":        event.frame_idx,
            "vehicle_a_id":     -1,
            "vehicle_b_id":     -1,
            "track_id":         event.track_id,
            "ttc_seconds":      -1.0,
            "distance_meters":  -1.0,
            "closing_speed_kmh": -1.0,
            "alert_level":      "N/A",
            "behavior_type":    event.behavior.value,
            "confidence":       round(event.confidence, 3),
            "severity":         event.severity,
            "description":      event.description
        }
        self._add_to_batch(row)

    def _add_to_batch(self, row: dict):
        """Add row to buffer and flush if batch size reached."""
        self.batch.append(row)
        if len(self.batch) >= self.batch_size:
            self.flush()

    def flush(self):
        """Insert all buffered rows into BigQuery."""
        if not self.batch:
            return

        errors = self.client.insert_rows_json(
            self.table_ref, self.batch
        )

        if errors:
            print(f"BigQuery insert errors: {errors}")
        else:
            self.total_logged += len(self.batch)
            print(
                f"BigQuery: inserted {len(self.batch)} rows "
                f"(total: {self.total_logged})"
            )

        self.batch = []

    def _alert_to_severity(self, alert_level: str) -> int:
        return {
            "SAFE": 1, "WARNING": 2,
            "HIGH": 3, "CRITICAL": 4
        }.get(alert_level, 1)

    def query_incidents(self, limit: int = 20) -> list:
        """
        Query the most recent incidents from BigQuery.
        Returns a list of row dicts.
        """
        sql = f"""
            SELECT
                event_type,
                alert_level,
                behavior_type,
                severity,
                ttc_seconds,
                distance_meters,
                description,
                timestamp
            FROM `{self.table_ref}`
            ORDER BY timestamp DESC
            LIMIT {limit}
        """
        rows = list(self.client.query(sql).result())
        return [dict(r) for r in rows]