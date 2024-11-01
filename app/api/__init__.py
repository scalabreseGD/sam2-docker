from functools import lru_cache

import toml

from .storage import FileUploader

__florence_serve = None
__file_uploader = None


def florence(model) -> Florence:
    global __florence_serve
    if __florence_serve is None:
        __florence_serve = FlorenceServe()
    return __florence_serve.get_or_load_model(model)


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
