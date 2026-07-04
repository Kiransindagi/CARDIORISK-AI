import logging
import sys
import uuid
from contextvars import ContextVar
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

request_id_ctx_var: ContextVar[str] = ContextVar("request_id", default="")

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [req_id=%(request_id)s] - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _get_extra(self):
        return {"request_id": request_id_ctx_var.get()}

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, extra=self._get_extra(), **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, extra=self._get_extra(), **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, extra=self._get_extra(), **kwargs)

def get_logger(name: str):
    return StructuredLogger(name)

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        token = request_id_ctx_var.set(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        request_id_ctx_var.reset(token)
        return response
