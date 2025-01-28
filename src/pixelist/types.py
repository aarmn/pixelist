from typing import Callable, List, Tuple, Union, Optional, Dict
from typing_extensions import Annotated
from pydantic import BaseModel, PlainValidator
from enum import Enum
from functools import wraps
import numpy as np
import inspect
from .features import check_feature

# Updated type definitions with proper validation
NumpyArray = Annotated[
    np.ndarray,
    PlainValidator(lambda x: x if isinstance(x, np.ndarray) else 
                   ValueError("Must be numpy array"))
]

ValidImageList = Annotated[
    List[NumpyArray],
    PlainValidator(lambda x: x if len(x) > 0 else 
                   ValueError("Image list cannot be empty"))
]

ValidName = Annotated[
    str,
    PlainValidator(lambda x: x if len(x) > 0 else 
                   ValueError("Name cannot be empty"))
]

class ProcessingStatus(str, Enum):
    INTERMEDIATE = "intermediate"
    FINAL = "final"

class ProcessingMode(str, Enum):
    NO_INTERMEDIATE = "no_intermediate"
    NO_INTERMEDIATE_SHOW_FINAL = "no_intermediate_show_final"
    WITH_INTERMEDIATE = "with_intermediate"
    WITH_INTERMEDIATE_SHOW_FINAL = "with_intermediate_show_final"
    WITH_INTERMEDIATE_SHOW_ALL = "with_intermediate_show_all"

class FilterDTO(BaseModel):
    """DTO for serializable representation of a Filter"""
    name: str
    description: Optional[str] = None
    function_name: str

class Filter(BaseModel):
    """
    Represents an image processing filter.
    
    Attributes:
        func (Callable): The actual filter function
        name (str): Name of the filter
        description (str, optional): Description of what the filter does
    """
    name: str
    description: Optional[str] = None
    func: Callable

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None):
        name = name or self.namer(func)
        super().__init__(func=func, name=name, description=description)
        # Preserve the function's metadata
        wraps(func)(self)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def to_dto(self) -> FilterDTO:
        return FilterDTO(
            name=self.name,
            description=self.description,
            function_name=self.func.__name__
        )

    @classmethod
    def from_dto(cls, dto: FilterDTO, func: Callable) -> 'Filter':
        return cls(func, name=dto.name, description=dto.description)

    def __str__(self) -> str:
        return self.name
    
    @staticmethod
    def namer(func: Callable) -> str:
        if func.__name__ == '<lambda>':
            frame = inspect.currentframe().f_back
            if frame:
                for var_name, var_val in frame.f_locals.items():
                    if var_val is func:
                        return var_name
            return f"unknown_filter_{id(func)}"
        else:
            return func.__name__

    @classmethod
    def make(cls, func: Callable, name: Optional[str] = None, description: Optional[str] = None):
        if not name:
            name = cls.namer(func)
        return cls(func, name=name, description=description)
    
    def __getattr__(self, name):
        # Delegate attribute access to wrapped function
        return getattr(self.func, name)
    
    def __dir__(self):
        # Include both wrapper and wrapped function attributes
        return sorted(set(super().__dir__() + dir(self.func)))

    def __repr__(self):
        return f"Filter({self.func.__name__}, name='{self.name}', description='{self.description}')"

    def dict(self, *args, **kwargs):
        # Custom dict method to handle serialization
        return self.to_dto().model_dump(*args, **kwargs)

# Update type definitions
FilterGroup = Union[Filter, List[Filter], Tuple[Filter, ...]]

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
        PlainValidator(lambda x: len(x) > 0 or ValueError("Must have at least one batch"))
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

class DisplayEntity(BaseModel):
    """Represents a single display entry with its images and history."""
    images: ValidImageList
    history: List[Filter]

    def history_str(self) -> str:
        """Convert filter history to string representation."""
        return "\n".join(str(f) for f in self.history) if self.history else "input"

class DisplayCollection(BaseModel):
    """Collection of display entities with their histories."""
    entities: Dict[str, DisplayEntity]

    @classmethod
    def from_processing_results(cls, results: List[ProcessingResult]) -> 'DisplayCollection':
        entities = {}
        for result in results:
            if isinstance(result.result, ImageBatch):
                entities[str(len(entities))] = DisplayEntity(
                    images=result.result.images,
                    history=result.result.history
                )
            else:  # ImageSuperposition
                for i, batch in enumerate(result.result.batches):
                    key = str(len(entities))
                    entities[key] = DisplayEntity(
                        images=batch.images,
                        history=batch.history
                    )
        return cls(entities=entities)

def _convert_results_to_display_dict(results: List[ProcessingResult]) -> DisplayCollection:
    """
    Convert processing results to a display collection.

    Args:
        results (List[ProcessingResult]): List of processing results

    Returns:
        DisplayCollection: Collection of display entities with their histories
    """
    return DisplayCollection.from_processing_results(results)

def _filter_history_to_string(history: List[Filter]) -> str:
    """Convert a list of filters to a string representation."""
    return "\n".join(str(f) for f in history)

class ImagePipeline:
    """
    A pipeline for processing images through a sequence of filters.

    The pipeline supports sequential and parallel processing of images through
    various filter combinations using both single filters and filter groups.

    Attributes:
        filters (List): Collection of filter functions or filter groups
        results (List[ProcessingResult]): Results from processing steps
    """

    def __init__(self, filters: Optional[FilterGroup] = None):
        """
        Initialize the pipeline with optional filters.

        Args:
            filters (Optional[FilterType]): Initial filters to add
        """
        self.filters = []
        self.results: List[ProcessingResult] = []
        if filters:
            if not isinstance(filters, (list, tuple)):
                filters = [filters]
            self.filters.extend(filters)

    def add_filter(self, filter_or_sequence: FilterGroup):
        self.filters.append(filter_or_sequence)
        return self

    def _process_batch(self, batch: ImageBatch, filter_obj: Filter) -> ImageBatch:
        """
        Process a single batch of images through a filter function.

        Args:
            batch (ImageBatch): Batch of images to process
            filter_obj (Filter): Filter function to apply

        Returns:
            ImageBatch: Processed batch with updated history
        """
        processed_images = [filter_obj(img) for img in batch.images]
        return ImageBatch(
            images=processed_images,
            history=batch.history + [filter_obj]
        )

    def _process_step(self,
                     current: Union[ImageBatch, ImageSuperposition],
                     step: FilterGroup,
                     is_final: bool = False) -> Union[ImageBatch, ImageSuperposition]:
        """
        Process a single step in the pipeline, handling both sequential and parallel processing.

        Args:
            current (Union[ImageBatch, ImageSuperposition]): Current state of images
            step (FilterType): Processing step to apply
            is_final (bool): Whether this is the final processing step

        Returns:
            Union[ImageBatch, ImageSuperposition]: Processed results
        """
        if isinstance(step, (list, tuple)):
            # For parallel processing, create multiple branches
            if isinstance(step, tuple):
                results = []
                for filter_obj in step:
                    if isinstance(current, ImageBatch):
                        result = self._process_batch(current, filter_obj)
                    else:  # ImageSuperposition
                        result = ImageSuperposition(
                            batches=[self._process_batch(batch, filter_obj)
                                   for batch in current.batches]
                        )
                    results.append(result)

                # Combine results into a superposition
                if all(isinstance(r, ImageBatch) for r in results):
                    return ImageSuperposition(batches=results)
                else:
                    return ImageSuperposition(
                        batches=[batch for result in results
                                for batch in result.batches]
                    )

            # For sequential processing, process each step in sequence
            else:  # list
                for substep in step:
                    current = self._process_step(current, substep)
                return current

        elif isinstance(step, Filter):
            if isinstance(current, ImageBatch):
                return self._process_batch(current, step)
            else:  # ImageSuperposition
                return ImageSuperposition(
                    batches=[self._process_batch(batch, step)
                            for batch in current.batches]
                )

        else:
            raise ValueError(f"Invalid step type: {type(step)}")

    def run(self, images: Union[np.ndarray, List[np.ndarray]],
            mode: ProcessingMode = ProcessingMode.WITH_INTERMEDIATE) -> List[ProcessingResult]:
        """
        Run the pipeline with specified processing mode.
        
        Args:
            images: Input images
            mode: ProcessingMode controlling intermediate results and display behavior
        """
        if isinstance(images, np.ndarray):
            images = [images]
        initial_batch = ImageBatch(images=images, history=[])

        self.results = [
            ProcessingResult(
                step_name='"input"',
                result=initial_batch,
                status=ProcessingStatus.INTERMEDIATE
            )
        ]

        current = initial_batch
        for i, step in enumerate(self.filters):
            is_final = i == len(self.filters) - 1
            result = self._process_step(current, step, is_final)

            if isinstance(result, ImageBatch):
                step_name = _filter_history_to_string(result.history)
            else:  # ImageSuperposition
                paths = [_filter_history_to_string(batch.history) for batch in result.batches]
                quoted_paths = [f'"{paths[0]}"'] + paths[1:]  # Quote first branch
                step_name = " | ".join(quoted_paths)

            store_this = (
                mode in (ProcessingMode.WITH_INTERMEDIATE, 
                        ProcessingMode.WITH_INTERMEDIATE_SHOW_ALL,
                        ProcessingMode.WITH_INTERMEDIATE_SHOW_FINAL)
                or is_final
            )

            if store_this:
                self.results.append(
                    ProcessingResult(
                        step_name=step_name,
                        result=result,
                        status=ProcessingStatus.FINAL if is_final
                               else ProcessingStatus.INTERMEDIATE
                    )
                )

            current = result

        # Handle display based on mode
        if mode in (ProcessingMode.NO_INTERMEDIATE_SHOW_FINAL, 
                   ProcessingMode.WITH_INTERMEDIATE_SHOW_FINAL):
            final_results = [r for r in self.results if r.status == ProcessingStatus.FINAL]
            display_collection = _convert_results_to_display_dict(final_results)
            display_images(display_collection)
        elif mode == ProcessingMode.WITH_INTERMEDIATE_SHOW_ALL:
            display_collection = _convert_results_to_display_dict(self.results)
            display_images(display_collection)

        return self.results

    @classmethod
    def make(cls,
             images: Union[np.ndarray, List[np.ndarray]],
             filters: FilterGroup,
             mode: ProcessingMode = ProcessingMode.WITH_INTERMEDIATE_SHOW_ALL) -> List[ProcessingResult]:
        """
        One-line convenience method to create pipeline, process images with specified mode.

        Args:
            images: Images to process
            filters: Filters to apply
            mode: ProcessingMode controlling intermediate results and display behavior

        Returns:
            List[ProcessingResult]: List of processing results
        """
        pipeline = cls(filters)
        return pipeline.run(images, mode=mode)

@check_feature("display", ["matplotlib", "cv2"])
def display_images(display_collection: DisplayCollection):
    """
    Display multiple images in a grid layout with proper labeling.

    Args:
        display_collection (DisplayCollection): Collection of images and their histories
    """
    import matplotlib.pyplot as plt
    import cv2

    image_dict = {
        entity.history_str(): entity.images
        for entity in display_collection.entities.values()
    }

    num_rows = len(image_dict)
    num_cols = len(list(image_dict.values())[0])
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, num_rows * 3))
    fig.suptitle('Image Processing Results', fontsize=16)

    if num_rows == 1:
        axes = axes.reshape(1, -1)

    row_index = 0
    for image_type, images in image_dict.items():
        for col_index, img in enumerate(images):
            if len(img.shape) == 3:
                axes[row_index, col_index].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                axes[row_index, col_index].imshow(img, cmap='gray')
            # Add more vertical space for multiline labels
            axes[row_index, col_index].set_title(f'{image_type}', pad=15)
            axes[row_index, col_index].axis('off')
        row_index += 1

    # Add more spacing between subplots for the multiline labels
    plt.tight_layout(h_pad=2.0)
    plt.show()
``` 

