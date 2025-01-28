from datasets import load_dataset
import numpy as np

def inspect_dataset():
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("aarmn/Persian_English_Roadsign_OCR_Dataset_Relabeled")
    
    # Print dataset structure
    print("\nDataset Structure:")
    print(f"Available splits: {dataset.keys()}")
    
    # Get train split info
    train_data = dataset['train']
    print(f"\nTrain split size: {len(train_data)}")
    
    # Get features info
    print("\nFeatures:")
    for feature in train_data.features:
        print(f"{feature}: {train_data.features[feature]}")
    
    # Sample one item to see actual data structure
    sample = train_data[0]
    print("\nSample item structure:")
    for key, value in sample.items():
        if hasattr(value, 'shape'):
            print(f"{key}: shape = {value.shape}, type = {type(value)}")
        else:
            print(f"{key}: type = {type(value)}")

if __name__ == "__main__":
    inspect_dataset()