from typing import Callable, List, Tuple, Union, Optional
from typing_extensions import Annotated
from pydantic import BaseModel, PlainValidator
from enum import Enum
from functools import wraps
import numpy as np
import inspect
import cv2
from urllib.request import urlopen

class Filter:
    """
    Represents an image processing filter.
    
    Attributes:
        func (Callable): The actual filter function
        name (str): Name of the filter
        description (str, optional): Description of what the filter does
    """
    def __init__(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None):
        self.func = func
        self.name = name or func.__name__
        self.description = description
        # Preserve the function's metadata
        wraps(func)(self)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

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

@Filter.make
def laplacian_filter(image: np.ndarray) -> np.ndarray:
    """Apply Laplacian filter for edge detection."""
    return cv2.Laplacian(image, cv2.CV_64F).astype(np.uint8)

def load_image_from_url(url: str) -> np.ndarray:
    """Load an image from a URL."""
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    return cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

def test_laplacian():
    # Load the standard Lena test image from the web
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    img = load_image_from_url(url)
    if img is None:
        raise RuntimeError("Failed to load image from URL")
    
    # Apply the filter
    edges = laplacian_filter(img)
    
    # Display results
    cv2.imshow('Original', img)
    cv2.imshow('Laplacian Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_laplacian()