import numpy as np
from .features import check_feature
from pydantic import BaseModel
from typing import List, Dict, Union, Optional
from .filter import Filter, FilterGroup
from .image import ImageBatch, ImageSuperposition, ValidImageList, ProcessingResult
from .enum import ProcessingMode, ProcessingStatus

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
            current: Current state of images (either batch or superposition)
            step: Processing step to apply (single filter or tuple of filters)
            is_final: Whether this is the final processing step

        Returns:
            Processed results as either ImageBatch or ImageSuperposition
        """
        # Handle tuple case (parallel processing)
        if isinstance(step, tuple):
            parallel_results = []
            
            # Process each filter in the tuple
            for filter_obj in step:
                if not isinstance(filter_obj, Filter):
                    raise ValueError(f"Expected Filter, got {type(filter_obj)}")
                    
                if isinstance(current, ImageBatch):
                    result = self._process_batch(current, filter_obj)
                else:  # ImageSuperposition
                    result = ImageSuperposition(
                        batches=[self._process_batch(batch, filter_obj) 
                                for batch in current.batches]
                    )
                parallel_results.append(result)
                
            # Ensure we have valid ImageBatch objects for superposition
            final_batches = []
            for result in parallel_results:
                if isinstance(result, ImageBatch):
                    final_batches.append(result)
                elif isinstance(result, ImageSuperposition):
                    final_batches.extend(result.batches)
                    
            return ImageSuperposition(batches=final_batches)
        
        # Handle single filter case
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