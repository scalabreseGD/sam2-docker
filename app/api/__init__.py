from functools import lru_cache

import toml

from .sam import SAM2, SAM2Serve
from .storage import FileUploader

__sam2_serve = None
__file_uploader = None


def sam2(model) -> SAM2:
    global __sam2_serve
    if __sam2_serve is None:
        __sam2_serve = SAM2Serve()
    return __sam2_serve.get_or_load_model(model)


def file_uploader() -> FileUploader:
    global __file_uploader
    if __file_uploader is None:
        conf = read_conf()
        __file_uploader = FileUploader(base_path=conf['file_uploader']['base_path'])
    return __file_uploader


@lru_cache(maxsize=1)
def read_conf():
    with open("conf/conf.toml", "r") as f:
        data = toml.load(f)
    return data
