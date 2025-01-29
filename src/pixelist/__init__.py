from importlib.metadata import version as _version
from .engine import (
    display_images,
    ImagePipeline
)
from .filter import Filter
from .image import (
    ImageBatch,
    ImageSuperposition, 
    ProcessingResult
)
from .enum import ProcessingStatus, ProcessingMode

# TODO: for showing add cv2 and plt, as feature
# TODO: Add typing

try:
    __version__ = _version("pixelist")
except ImportError:
    # Package is not installed
    __version__ = "unknown"

__all__ = [
    "display_images",
    "ImageBatch", 
    "ImagePipeline",
    "ImageSuperposition",
    "ProcessingResult",
    "Filter",
    "ProcessingStatus",
    "ProcessingMode",
]