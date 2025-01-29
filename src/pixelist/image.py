from typing import Annotated, List, Union

from numpy import ndarray
from pydantic import BaseModel, PlainValidator

from .enum import ProcessingStatus
from .filter import Filter

NumpyArray = Annotated[
    ndarray,
    PlainValidator(lambda x: x if isinstance(x, ndarray) else ValueError("Must be numpy array")),
]

ValidImageList = Annotated[
    List[NumpyArray],
    PlainValidator(lambda x: x if len(x) > 0 else ValueError("Image list cannot be empty")),
]

ValidName = Annotated[
    str, PlainValidator(lambda x: x if len(x) > 0 else ValueError("Name cannot be empty"))
]


class ImageBatch(BaseModel):
    """
    A container for a batch of images and their processing history.

    Attributes:
        images (ValidImageList): List of numpy arrays representing images
        history (List[Filter]): List of filter functions applied to the images
    """

    images: ValidImageList
    history: List[Filter] = []


class ImageSuperposition(BaseModel):
    """
    A collection of multiple ImageBatches representing parallel processing branches.

    Attributes:
        batches (List[ImageBatch]): List of image batches from different processing paths
    """

    batches: Annotated[
        List[ImageBatch],
        PlainValidator(
            lambda x: x
            if isinstance(x, list) and len(x) > 0 and all(isinstance(b, ImageBatch) for b in x)
            else ValueError("Must be non-empty list of ImageBatch objects")
        ),
    ]


class ProcessingResult(BaseModel):
    """
    Represents the result of a processing step in the pipeline.

    Attributes:
        step_name (ValidName): Name of the processing step
        result (Union[ImageBatch, ImageSuperposition]): Output of the processing step
        status (ProcessingStatus): Indicates if the result is intermediate or final
    """

    step_name: ValidName
    result: Union[ImageBatch, ImageSuperposition]
    status: ProcessingStatus
