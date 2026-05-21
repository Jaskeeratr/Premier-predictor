from __future__ import annotations


class AppError(Exception):
    status_code = 400

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code


class ValidationError(AppError):
    status_code = 400


class ExternalServiceError(AppError):
    status_code = 502
