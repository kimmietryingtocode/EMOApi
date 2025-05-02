import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class LoggingMiddelware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        print(
            f"[{request.method}] {request.url.path} - {response.status_code} - {duration:.3f}s"
        )
        return response
