from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from api import read_conf, file_uploader


class LimitRequestSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        conf = read_conf()
        self.max_upload_size = conf['app']['max_upload_size']

    async def dispatch(self, request: Request, call_next):
        if request.headers.get('content-type') and 'multipart' in request.headers.get('content-type'):
            response = await call_next(request)
            return response

        # Check if content length exceeds the max size
        if request.headers.get("content-length"):
            content_length = int(request.headers["content-length"])
            if content_length > self.max_upload_size:
                return Response(
                    "Request body is too large", status_code=413
                )

        response = await call_next(request)
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    file_uploader().delete_mount()
