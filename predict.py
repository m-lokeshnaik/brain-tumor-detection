import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import argparse
from pathlib import Path
from train import ModifiedResNet50

def load_model(model_path, device):
    """Load the trained model."""
    model = ModifiedResNet50(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    """Preprocess the input image for prediction."""
    # Read and preprocess image
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Normalize intensity
    image = image.astype(np.float32) / 255.0
    
    # Resize
    image = cv2.resize(image, (224, 224))
    
    # Convert to 3 channels
    image = np.stack([image] * 3, axis=-1)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, device):
    """Make prediction on the input image."""
    classes = ['normal', 'meningioma', 'glioma', 'pituitary']
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get prediction and confidence
        pred_prob, pred_class = torch.max(probabilities, dim=1)
        prediction = classes[pred_class.item()]
        confidence = pred_prob.item() * 100
        
        # Get all class probabilities
        class_probs = {cls: prob * 100 for cls, prob in zip(classes, probabilities[0].cpu().numpy())}
        
        return prediction, confidence, class_probs

def main():
    parser = argparse.ArgumentParser(description='Predict brain tumor from MRI scan')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        # Load model
        model = load_model(args.model_path, device)
        
        # Preprocess image
        image_tensor = preprocess_image(args.input)
        
        # Make prediction
        prediction, confidence, class_probs = predict(model, image_tensor, device)
        
        # Print results
        print("\nPrediction Results:")
        print(f"Predicted Class: {prediction}")
        print(f"Confidence: {confidence:.2f}%")
        
        print("\nClass Probabilities:")
        for cls, prob in class_probs.items():
            print(f"{cls}: {prob:.2f}%")
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

if __name__ == '__main__':
    main() 