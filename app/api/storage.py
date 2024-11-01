import os
import shutil

from fastapi import HTTPException
from pathlib import Path


class FileUploader:
    def __init__(self, base_path):
        pid = str(os.getpid())
        # home = Path.home()
        self.folder = os.path.join(base_path, pid)
        self.__create_folder(self.folder)

    @staticmethod
    def __create_folder(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def upload(self, file):
        if file.content_type not in ["image/jpeg", "image/png", "video/mp4"]:
            raise HTTPException(status_code=400, detail="Only JPEG, PNG images and MP4 videos are allowed.")
        file_path = os.path.join(self.folder, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return file_path

    def upload_batch(self, files):
        return [self.upload(file) for file in files]

    def delete_mount(self):
        shutil.rmtree(self.folder)
