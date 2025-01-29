from typing import Callable, List, Tuple, Union, Optional
from pydantic import BaseModel
from functools import wraps
import inspect


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

    def __init__(
        self, func: Callable, name: Optional[str] = None, description: Optional[str] = None
    ):
        name = name or self.namer(func)
        super().__init__(func=func, name=name, description=description)
        # Preserve the function's metadata
        wraps(func)(self)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def to_dto(self) -> FilterDTO:
        return FilterDTO(
            name=self.name, description=self.description, function_name=self.func.__name__
        )

    @classmethod
    def from_dto(cls, dto: FilterDTO, func: Callable) -> "Filter":
        return cls(func, name=dto.name, description=dto.description)

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def namer(func: Callable) -> str:
        if func.__name__ == "<lambda>":
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
