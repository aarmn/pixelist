from importlib.util import find_spec
from functools import wraps

def has_module(module_name: str) -> bool:
    return find_spec(module_name) is not None

def check_feature(feature_name: str, required_modules: list[str]):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            missing = [mod for mod in required_modules if not has_module(mod)]
            if missing:
                raise ImportError(
                    f"Feature '{feature_name}' requires {missing}. "
                    f"Install with: pip install pixelist[{feature_name}]"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator