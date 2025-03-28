import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from concurrent.futures import ThreadPoolExecutor

def create_preprocessing_pipeline():
    """Create the preprocessing pipeline using albumentations."""
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485], std=[0.229]),
        A.OneOf([
            A.RandomRotate90(),
            A.Rotate(limit=15),
        ], p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    ])

def normalize_intensity(image):
    """Normalize image intensity to 0-1 range."""
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val == 0:
        return image
    return (image - min_val) / (max_val - min_val)

def skull_stripping(image):
    """Basic skull stripping using Otsu thresholding and morphological operations."""
    # Convert to uint8 if not already
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Apply mask to original image
    return cv2.bitwise_and(image, image, mask=mask)

def process_image(args):
    """Process a single image with all preprocessing steps."""
    input_path, output_path, transform = args
    
    # Read image
    image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to read image: {input_path}")
        return
    
    # Apply preprocessing steps
    image = normalize_intensity(image)
    image = skull_stripping(image)
    
    # Apply augmentation pipeline
    transformed = transform(image=image)
    processed_image = transformed['image']
    
    # Save processed image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(str(output_path), processed_image)

def main():
    parser = argparse.ArgumentParser(description='Prepare MRI data for training')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing raw images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed images')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create preprocessing pipeline
    transform = create_preprocessing_pipeline()

    # Collect all image files
    input_dir = Path(args.input_dir)
    image_files = list(input_dir.glob('**/*.jpg')) + list(input_dir.glob('**/*.png'))
    
    # Prepare processing arguments
    process_args = []
    for input_path in image_files:
        relative_path = input_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        process_args.append((input_path, output_path, transform))

    # Process images in parallel
    print(f"Processing {len(image_files)} images...")
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        list(tqdm(executor.map(process_image, process_args), total=len(process_args)))

    print("Data preparation completed!")

if __name__ == '__main__':
    main() 