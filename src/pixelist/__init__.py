from importlib.metadata import version as _version
from .types import (
    display_images,
    filter_decorator,
    ImageBatch,
    ImagePipeline,
    ImageSuperposition,
    ProcessingResult,
)

# TODO: for showing add cv2 and plt, as feature
# TODO: Add typing

try:
    __version__ = _version("pixelist")
except ImportError:
    # Package is not installed
    __version__ = "unknown"

__all__ = [
    "display_images",
    "filter_decorator",
    "ImageBatch", 
    "ImagePipeline",
    "ImageSuperposition",
    "ProcessingResult",
]