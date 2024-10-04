from copy import deepcopy
from pathlib import Path
from typing import Generator, Union

from google.cloud import storage
from google.cloud.storage.bucket import Bucket


# Google Cloud Storage の特定のバケットのオブジェクト操作（アップロード，削除）をするクラス
# オブジェクトの削除は，このクラスを用いてアップロードされたものにのみ適応可能
class GCSBucket:
    def __init__(self, bucket_name: str, client: storage.Client =None) -> 'GCSBucket':
        if client is None:
            client = storage.Client()

        bucket = client.lookup_bucket(bucket_name)
        assert isinstance(bucket, Bucket), f"bucket {bucket_name} does not exist"

        self.__bucket = bucket
        self._uploaded_files = []

    @property
    def uploaded_files(self) -> None:
        return deepcopy(self._uploaded_files)

    @classmethod
    def from_json(cls, bucket_name: str, json_path: Union[str, Path]) -> 'GCSBucket':
        client = storage.Client.from_service_account_json(json_path)

        return cls(bucket_name, client=client)

    def __str__(self) -> str:
        return self.__bucket.name

    def __get_blob(self, filename: str) -> storage.Blob:
        blob = self.__bucket.blob(filename)

        assert blob.exists(), f"File {filename} does not exist in bucket {self.__bucket.name}"

        return blob

    def upload(self, filename: str) -> None:
        try:
            self.__get_blob(filename)
        except Exception:
            blob = self.__bucket.blob(filename)
            blob.upload_from_filename(filename)

            self._uploaded_files.append(filename)
        else:
            print(f"File {filename} already exists in {self.__bucket.name}")

    def delete(self, filename: str) -> None:
        if filename in self._uploaded_files:
            blob = self.__get_blob(filename)
            blob.delete()

            self._uploaded_files.remove(filename)
        else:
            print(f"Can't delete File {filename}; you are not allowed to delete File uploaded by others")

    def get_gcs_uri(self, filename: str) -> str:
        self.__get_blob(filename)

        uri = f"gs://{self.__bucket.name}/{filename}"

        return uri

    def gcs_uri_generator(self, **kwargs) -> Generator[str, None, None]:
        for blob in self.__bucket.list_blobs(**kwargs):
            uri = f"gs://{self.__bucket.name}/{blob.name}"

            yield uri
