"""
Storage Service - Optional Google Cloud Storage
Always fails silently - never crashes the app
"""

import os
import base64
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class StorageService:
    """
    Optional GCS storage. If GCS is not configured or unavailable,
    all methods return None silently. The app works fine without it.
    """

    def __init__(self):
        self.client = None
        self.bucket = None

        # Only attempt GCS if project ID is explicitly set
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
        if not project_id:
            logger.info("StorageService: running without GCS (local mode)")
            return

        try:
            from google.cloud import storage as gcs
            self.client = gcs.Client(project=project_id)
            bucket_name = os.getenv("GCS_BUCKET_NAME", f"{project_id}-tutor-uploads")
            self.bucket = self.client.bucket(bucket_name)
            logger.info(f"StorageService: GCS connected ({bucket_name})")
        except Exception:
            # Silently disable GCS - never crash
            self.client = None
            self.bucket = None
            logger.info("StorageService: GCS unavailable, using local mode")

    async def upload_frame(self, session_id: str, image_base64: str) -> Optional[str]:
        if not self.client:
            return None
        try:
            data = base64.b64decode(image_base64)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            blob = self.bucket.blob(f"frames/{session_id}/{ts}.jpg")
            blob.upload_from_string(data, content_type="image/jpeg")
            return f"gs://{self.bucket.name}/frames/{session_id}/{ts}.jpg"
        except Exception:
            return None

    async def upload_diagram(self, session_id: str, svg: str) -> Optional[str]:
        if not self.client:
            return None
        try:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            blob = self.bucket.blob(f"diagrams/{session_id}/{ts}.svg")
            blob.upload_from_string(svg.encode(), content_type="image/svg+xml")
            blob.make_public()
            return blob.public_url
        except Exception:
            return None
