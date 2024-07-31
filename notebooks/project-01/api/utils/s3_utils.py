import logging
import os
from typing import Dict

import boto3
from botocore.exceptions import ClientError
from config import Config

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class S3Utils:
    def __init__(self, config: Config):
        """
        Initialize the S3Utils with configuration.

        Args:
            config (Config): Configuration object containing S3 settings.
        """
        self.config = config
        self.bucket_name = config["S3_BUCKET_NAME"]
        self.client = boto3.client(
            "s3",
            endpoint_url=config["S3_ENDPOINT_URL"],
            aws_access_key_id=config["S3_ACCESS_KEY_ID"],
            aws_secret_access_key=config["S3_SECRET_ACCESS_KEY"],
            region_name=config["S3_REGION_NAME"],
        )

    def create_bucket(self):
        """
        Create an S3 bucket if it doesn't exist.
        """
        try:
            self.client.create_bucket(Bucket=self.bucket_name)
            print(f"Created S3 bucket: {self.bucket_name}")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "BucketAlreadyOwnedByYou":
                print(f"S3 bucket already exists: {self.bucket_name}")
            else:
                print(f"Error creating S3 bucket: {str(e)}")

    def upload_file(self, file_path: str, object_name: str = None):
        """
        Upload a file to the S3 bucket.

        Args:
            file_path (str): Path to the file to upload.
            object_name (str, optional): S3 object name. If not specified, the file name is used.
        """
        if object_name is None:
            object_name = os.path.basename(file_path)

        try:
            self.client.upload_file(file_path, self.bucket_name, object_name)
            print(f"Uploaded {file_path} to S3 bucket {self.bucket_name}")
        except ClientError as e:
            print(f"Error uploading to S3: {str(e)}")

    def download_file(self, object_name: str, file_path: str):
        try:
            logger.debug(f"Attempting to download {object_name} to {file_path}")
            logger.debug(f"Using endpoint: {self.client.meta.endpoint_url}")
            self.client.download_file(self.bucket_name, object_name, file_path)
            logger.debug(f"Downloaded {object_name} from S3 bucket {self.bucket_name}")
        except ClientError as e:
            logger.error(f"Error downloading from S3: {str(e)}")
            raise

    def list_objects(self, prefix: str = ""):
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=prefix
            )
            objects = [obj["Key"] for obj in response.get("Contents", [])]
            logger.debug(f"Objects in bucket {self.bucket_name}: {objects}")
            return objects
        except ClientError as e:
            logger.error(f"Error listing objects in S3: {str(e)}")
            return []

    def delete_object(self, object_name: str):
        """
        Delete an object from the S3 bucket.

        Args:
            object_name (str): Name of the object to delete.
        """
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=object_name)
            print(f"Deleted {object_name} from S3 bucket {self.bucket_name}")
        except ClientError as e:
            print(f"Error deleting object from S3: {str(e)}")

    def upload_model_artifacts(self, model_dir: str):
        """
        Upload model artifacts to S3.

        Args:
            model_dir (str): Directory containing model artifacts.
        """
        artifacts = [
            "best_model.joblib",
            "feature_preprocessor.joblib",
            "target_preprocessor.joblib",
            "model_comparison.png",
        ]
        for artifact in artifacts:
            file_path = os.path.join(model_dir, artifact)
            if os.path.exists(file_path):
                self.upload_file(file_path, artifact)
            else:
                print(f"Warning: {artifact} not found in {model_dir}")

    def upload_performance_plot(self, plot_path: str):
        """
        Upload the performance plot to S3.

        Args:
            plot_path (str): Path to the performance plot file.
        """
        object_name = "model_comparison.png"
        try:
            self.upload_file(plot_path, object_name)
            print(f"Uploaded performance plot to S3: {object_name}")
        except Exception as e:
            print(f"Error uploading performance plot to S3: {str(e)}")

    def load_model_artifacts(self, model_dir: str):
        artifacts = [
            "best_model.joblib",
            "feature_preprocessor.joblib",
            "target_preprocessor.joblib",
            "model_comparison.png",
        ]
        for artifact in artifacts:
            file_path = os.path.join(model_dir, artifact)
            try:
                self.download_file(artifact, file_path)
            except ClientError:
                print(f"Warning: {artifact} not found in S3 bucket {self.bucket_name}")
