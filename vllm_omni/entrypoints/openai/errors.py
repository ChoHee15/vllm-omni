class InvalidInputReferenceError(ValueError):
    def __init__(self, message: str = "Invalid input reference.") -> None:
        super().__init__(message)


class InputReferenceTooLargeError(ValueError):
    """Raised when a decoded input reference exceeds the configured size limit.

    Callers should surface this as HTTP 413 (Request Entity Too Large).
    """

    def __init__(self, message: str = "Input reference exceeds the size limit.") -> None:
        super().__init__(message)
