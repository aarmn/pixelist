import numpy as np
import pytest

from pixelist import Filter, ImageBatch, ImagePipeline, ImageSuperposition, ProcessingMode


@pytest.fixture
def sample_image():
    """Create a sample 10x10 grayscale image."""
    return np.random.randint(0, 255, (10, 10), dtype=np.uint8)

@pytest.fixture
def sample_filter():
    @Filter.make
    def identity(image):
        return image
    return identity

def test_filter_creation():
    """Test Filter creation and decoration."""
    @Filter.make
    def test_filter(image):
        return image
    
    assert isinstance(test_filter, Filter)
    assert test_filter.name == "test_filter"

def test_pipeline_sequential(sample_image, sample_filter):
    """Test sequential pipeline processing."""
    pipeline = ImagePipeline([sample_filter])
    results = pipeline.run([sample_image], ProcessingMode.NO_INTERMEDIATE)
    
    assert len(results) == 1
    assert isinstance(results[0].result, ImageBatch)

def test_pipeline_parallel(sample_image, sample_filter):
    """Test parallel pipeline processing."""
    pipeline = ImagePipeline([(sample_filter, sample_filter)])
    results = pipeline.run([sample_image], ProcessingMode.NO_INTERMEDIATE)
    
    assert len(results) == 1
    assert isinstance(results[0].result, ImageSuperposition)
    assert len(results[0].result.batches) == 2

def test_empty_pipeline():
    """Test pipeline creation with no filters."""
    with pytest.raises(ValueError):
        ImagePipeline([])

def test_invalid_image_input(sample_filter):
    """Test pipeline with invalid image input."""
    pipeline = ImagePipeline([sample_filter])
    with pytest.raises(ValueError):
        pipeline.run([])

def test_processing_modes(sample_image, sample_filter):
    """Test different processing modes."""
    pipeline = ImagePipeline([sample_filter])

    # Test NO_INTERMEDIATE
    results = pipeline.run([sample_image], ProcessingMode.NO_INTERMEDIATE)
    assert len(results) == 1

    # Test WITH_INTERMEDIATE
    results = pipeline.run([sample_image], ProcessingMode.WITH_INTERMEDIATE)
    assert len(results) == 2