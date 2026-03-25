import json
import os
import time
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class IncidentPublisher:
    """
    Streams highway safety incidents to GCP Pub/Sub in real time.

    Every near-miss, tailgating event, sudden brake, or wrong-way
    detection gets published as a JSON message to the topic.
    Downstream systems (dashboard, alerting, BigQuery) subscribe
    and react instantly.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        topic_id: Optional[str] = None,
        credentials_path: Optional[str] = None
    ):
        self.project_id = project_id or os.getenv(
            'GCP_PROJECT_ID', 'highway-safety-ai-jude'
        )
        self.topic_id = topic_id or os.getenv(
            'PUBSUB_TOPIC', 'highway-safety-events'
        )
        credentials_path = credentials_path or os.getenv(
            'GOOGLE_APPLICATION_CREDENTIALS', './secrets/gcp-key.json'
        )

        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

        from google.cloud import pubsub_v1
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(
            self.project_id, self.topic_id
        )

        # Ensure topic exists
        self._ensure_topic()

        # Stats
        self.published_count = 0
        self.failed_count = 0

        print(f"IncidentPublisher ready → {self.topic_path}")

    def _ensure_topic(self):
        """Create Pub/Sub topic if it doesn't exist."""
        from google.cloud import pubsub_v1
        from google.api_core.exceptions import AlreadyExists

        admin = pubsub_v1.PublisherClient()
        try:
            admin.create_topic(request={"name": self.topic_path})
            print(f"Created topic: {self.topic_id}")
        except AlreadyExists:
            print(f"Topic exists: {self.topic_id} ✅")

    def publish_near_miss(
        self,
        event,          # NearMissEvent from near_miss.py
        sequence_id: str,
        camera_id: str = "UA-DETRAC-CAM"
    ) -> bool:
        """Publish a near-miss event to Pub/Sub."""
        payload = {
            "event_type":       "near_miss",
            "timestamp":        datetime.utcnow().isoformat() + "Z",
            "sequence_id":      sequence_id,
            "camera_id":        camera_id,
            "frame_idx":        event.frame_idx,
            "vehicle_a_id":     event.vehicle_a_id,
            "vehicle_b_id":     event.vehicle_b_id,
            "ttc_seconds":      round(event.ttc_seconds, 3),
            "distance_meters":  round(event.distance_meters, 2),
            "closing_speed_kmh": round(event.closing_speed_kmh, 1),
            "alert_level":      event.alert_level.value,
            "severity":         self._alert_to_severity(
                                    event.alert_level.value
                                ),
            "description":      event.description
        }
        return self._publish(payload, event_type="near_miss")

    def publish_behavior(
        self,
        event,          # BehaviorEvent from behavior_classifier.py
        sequence_id: str,
        camera_id: str = "UA-DETRAC-CAM"
    ) -> bool:
        """Publish a behavior event to Pub/Sub."""
        payload = {
            "event_type":    "behavior",
            "timestamp":     datetime.utcnow().isoformat() + "Z",
            "sequence_id":   sequence_id,
            "camera_id":     camera_id,
            "frame_idx":     event.frame_idx,
            "track_id":      event.track_id,
            "behavior_type": event.behavior.value,
            "confidence":    round(event.confidence, 3),
            "severity":      event.severity,
            "description":   event.description
        }
        return self._publish(payload, event_type="behavior")

    def publish_frame_summary(
        self,
        frame_idx: int,
        vehicle_count: int,
        incident_count: int,
        sequence_id: str,
        camera_id: str = "UA-DETRAC-CAM"
    ) -> bool:
        """Publish per-frame summary for dashboard heartbeat."""
        payload = {
            "event_type":     "frame_summary",
            "timestamp":      datetime.utcnow().isoformat() + "Z",
            "sequence_id":    sequence_id,
            "camera_id":      camera_id,
            "frame_idx":      frame_idx,
            "vehicle_count":  vehicle_count,
            "incident_count": incident_count
        }
        return self._publish(payload, event_type="frame_summary")

    def _publish(self, payload: dict, event_type: str) -> bool:
        """
        Publish a JSON payload to Pub/Sub.
        Returns True on success, False on failure.
        """
        try:
            data = json.dumps(payload).encode("utf-8")

            # Attributes allow filtering on the subscription side
            attributes = {
                "event_type": event_type,
                "severity":   str(payload.get("severity", 1))
            }

            future = self.publisher.publish(
                self.topic_path,
                data=data,
                **attributes
            )

            # Wait for confirmation (non-blocking in production,
            # blocking here to catch errors during dev)
            message_id = future.result(timeout=10)
            self.published_count += 1
            return True

        except Exception as e:
            self.failed_count += 1
            print(f"Publish failed [{event_type}]: {e}")
            return False

    def _alert_to_severity(self, alert_level: str) -> int:
        return {
            "SAFE":     1,
            "WARNING":  2,
            "HIGH":     3,
            "CRITICAL": 4
        }.get(alert_level, 1)

    def flush(self):
        """Wait for all pending publishes to complete."""
        self.publisher.transport._channel.close()

    def get_stats(self) -> dict:
        return {
            "published": self.published_count,
            "failed":    self.failed_count,
            "topic":     self.topic_path
        }