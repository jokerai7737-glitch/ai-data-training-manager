"""Application exceptions."""


class PipelineError(Exception):
    """Base exception for the application."""


class ValidationError(PipelineError):
    """Raised when input data is invalid."""


class RepositoryError(PipelineError):
    """Raised when storage operations fail."""


class TrainingError(PipelineError):
    """Raised when training cannot be completed."""
