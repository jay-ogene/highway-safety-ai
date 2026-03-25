import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class ModelUploader:
    """
    Uploads and downloads YOLO model weights to/from GCS.

    Keeps best.pt in cloud storage so any machine can
    pull the latest model without manual file transfers.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        bucket_name: Optional[str] = None,
        credentials_path: Optional[str] = None
    ):
        self.project_id = project_id or os.getenv(
            'GCP_PROJECT_ID', 'highway-safety-ai-jude'
        )
        self.bucket_name = bucket_name or os.getenv(
            'GCS_MODELS_BUCKET', 'highway-safety-models-jude'
        )
        credentials_path = credentials_path or os.getenv(
            'GOOGLE_APPLICATION_CREDENTIALS', './secrets/gcp-key.json'
        )

        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

        from google.cloud import storage
        self.client = storage.Client(project=self.project_id)
        self.bucket = self.client.bucket(self.bucket_name)

        print(f"ModelUploader ready → gs://{self.bucket_name}")

    def upload_model(
        self,
        local_path: str,
        model_name: str = "best_yolov8s_v1.pt",
        version_tag: Optional[str] = None
    ) -> str:
        """
        Upload a model weights file to GCS.
        Returns the GCS URI.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Model not found: {local_path}")

        # Build GCS path
        tag = version_tag or "latest"
        gcs_path = f"models/{tag}/{model_name}"

        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(str(local_path))

        size_mb = local_path.stat().st_size / 1e6
        gcs_uri = f"gs://{self.bucket_name}/{gcs_path}"
        print(f"Uploaded: {local_path.name} ({size_mb:.1f}MB) → {gcs_uri}")
        return gcs_uri

    def download_model(
        self,
        local_path: str,
        model_name: str = "best_yolov8s_v1.pt",
        version_tag: str = "latest"
    ) -> str:
        """
        Download a model from GCS to local path.
        Returns local path string.
        """
        gcs_path = f"models/{version_tag}/{model_name}"
        blob = self.bucket.blob(gcs_path)

        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_path)

        size_mb = Path(local_path).stat().st_size / 1e6
        print(f"Downloaded: {gcs_path} ({size_mb:.1f}MB) → {local_path}")
        return local_path

    def list_models(self) -> list:
        """List all model files in the bucket."""
        blobs = self.client.list_blobs(
            self.bucket_name, prefix="models/"
        )
        models = []
        for blob in blobs:
            models.append({
                "name":     blob.name,
                "size_mb":  round(blob.size / 1e6, 1),
                "updated":  blob.updated.isoformat()
            })
            print(f"  {blob.name} ({blob.size/1e6:.1f}MB)")
        return models