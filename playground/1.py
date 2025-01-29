from datasets import load_dataset
import numpy as np
from pixelist import ImagePipeline, ImageBatch, filter_decorator
import cv2

from datasets import load_dataset
import numpy as np
from PIL import Image
import cv2

def load_sample_images(num_samples: int = 5) -> list[np.ndarray]:
    """Load sample images from the road sign dataset."""
    dataset = load_dataset("aarmn/Persian_English_Roadsign_OCR_Dataset_Relabeled")
    
    # Get training split and sample randomly
    train_data = dataset['train']
    # Convert numpy.int32 to regular Python int
    indices = [int(i) for i in np.random.choice(len(train_data), num_samples, replace=False)]
    
    # Convert images to numpy arrays
    images = []
    for idx in indices:
        # Get PIL Image from dataset
        img: Image.Image = train_data[idx]['image']
        # Convert PIL to numpy array
        img_array = np.array(img)
        # Convert to grayscale if colored (RGB to GRAY)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        images.append(img_array)
    
    return images

def laplacian_filter(image: np.ndarray) -> np.ndarray:
    """Apply Laplacian filter for edge detection."""
    return cv2.Laplacian(image, cv2.CV_64F).astype(np.uint8)

def prewitt_filter(image: np.ndarray) -> np.ndarray:
    """Apply Prewitt filter for edge detection."""
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    grad_x = cv2.filter2D(image, -1, kernel_x)
    grad_y = cv2.filter2D(image, -1, kernel_y)
    
    return cv2.addWeighted(np.absolute(grad_x), 0.5, 
                          np.absolute(grad_y), 0.5, 0)

def histogram_stretch(image: np.ndarray) -> np.ndarray:
    """Apply histogram stretching."""
    p2, p98 = np.percentile(image, (2, 98))
    return np.clip((image - p2) * (255.0 / (p98 - p2)), 0, 255).astype(np.uint8)

def main():
    # Load sample images
    print("Loading dataset samples...")
    images = load_sample_images(4)
    
    # Add parentheses to decorator calls
    histogram_stretch_filter = filter_decorator()(histogram_stretch)
    prewitt_filter_decorated = filter_decorator()(prewitt_filter)
    laplacian_filter_decorated = filter_decorator()(laplacian_filter)

    # Create sequential pipeline
    seq_pipeline = ImagePipeline([histogram_stretch_filter, prewitt_filter_decorated, laplacian_filter_decorated])
    seq_result = seq_pipeline.run(images=images, show=True)
    
    # Create parallel pipeline
    par_pipeline = ImagePipeline([histogram_stretch_filter, (prewitt_filter_decorated, laplacian_filter_decorated)])
    par_result = par_pipeline.run(images=images, show=True)
    
    print(dir(par_result))
    print(dir(seq_result))

if __name__ == "__main__":
    main()
    print("All tests completed successfully!")