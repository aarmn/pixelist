# Pixelist 🎨

[![PyPI version](https://badge.fury.io/py/pixelist.svg)](https://pypi.org/project/pixelist/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful, composable image processing pipeline library for Python that makes working with image filters fun and flexible! 🚀

## ✨ Features

- 🔄 Sequential and parallel image processing pipelines
- 🎯 Easy-to-use filter composition
- 📊 Built-in visualization support
- 🛡️ Type-safe with Pydantic validation
- 🎨 Support for both grayscale and color images
- 📝 Comprehensive processing history tracking

## 🚀 Installation

```bash
# Basic installation
pip install pixelist

# With visualization support
pip install pixelist[display]
```

## 🎯 Quick Start

Here's a simple example to get you started:

```python
from pixelist import ImagePipeline, Filter, ProcessingMode
import numpy as np

# Define some filters
@Filter.make
def threshold(image: np.ndarray) -> np.ndarray:
    return np.where(image > 127, 255, 0)

@Filter.make
def blur(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (5, 5), 0)

# Create and run a pipeline
pipeline = ImagePipeline([blur, threshold])
results = pipeline.run(
    images=your_image,
    mode=ProcessingMode.WITH_INTERMEDIATE_SHOW_ALL
)
```

## 🌟 Advanced Usage

### Parallel Processing

```python
# Create parallel branches in your pipeline
pipeline = ImagePipeline([
    histogram_stretch,
    (prewitt_filter, laplacian_filter)  # Parallel filters
])
```

### Custom Filter Creation

```python
@Filter.make
def my_awesome_filter(image: np.ndarray) -> np.ndarray:
    # Your image processing magic here
    return processed_image
```

## 🎨 Visualization

The library includes built-in visualization support:

```python
from pixelist import ImagePipeline, ProcessingMode

pipeline.run(
    images=input_images,
    mode=ProcessingMode.WITH_INTERMEDIATE_SHOW_ALL  # Shows all steps
)
```

## 🛠️ Processing Modes

- `NO_INTERMEDIATE`: Just the final result
- `NO_INTERMEDIATE_SHOW_FINAL`: Show final result visually
- `WITH_INTERMEDIATE`: Keep all intermediate results
- `WITH_INTERMEDIATE_SHOW_ALL`: Visual display of all steps
- `WITH_INTERMEDIATE_SHOW_FINAL`: Keep all, show final

<!-- ## 📚 Documentation

For more examples and detailed documentation, check out our [documentation](https://github.com/yourusername/pixelist/docs). -->

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Open issues
- Submit PRs
- Suggest improvements
- Share the love ❤️

## 📝 License

MIT License - feel free to use in your projects!

## 🙏 Acknowledgments

Special thanks to:
- The NumPy and OpenCV communities
- All our contributors

---

Made with ❤️ by the AARMN The Limitless

Remember to ⭐ the repo if you like it!
