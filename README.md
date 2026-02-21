# AIP - Adversarial Image Perturbations

A web app that demonstrates adversarial attacks on AI image recognition models using FGSM (Fast Gradient Sign Method).

## What It Does

- **Generate Adversarial Images**: Adds imperceptible noise to images to fool AI models
- **Show Perturbations**: Visualizes the noise patterns added to images
- **Compare Predictions**: Shows original vs adversarial predictions side-by-side
- **Detect Adversarial Images**: Analyzes if images contain perturbations
- **Educational Tool**: Learn how AI models can be fooled

## How It Works

1. Upload an image (PNG, JPG, JPEG)
2. Select attack intensity (epsilon value: 0.01 - 0.3)
3. AI generates adversarial version with minimal noise
4. Original model prediction changes on adversarial image
5. View original, perturbation, and adversarial images

## Tech Stack

- **Backend**: Flask
- **AI Model**: MobileNetV2 (ImageNet pre-trained)
- **Attack Method**: FGSM (Fast Gradient Sign Method)
- **Image Processing**: OpenCV, PIL, TensorFlow
- **Frontend**: HTML/CSS/JavaScript

## View
[LIVE DEMO](https://adversarial-shield.onrender.com/)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

# Open http://localhost:5000
```

## Features

- FGSM adversarial attack generation
- Real-time image processing
- Adversarial detection using Laplacian variance
- Batch processing with concurrent execution
- Detailed prediction confidence scores
- Interactive epsilon adjustment

## Attack Intensity Levels

- **Low (0.01-0.05)**: Subtle noise, harder to detect
- **Medium (0.1-0.15)**: Moderate noise, visible changes
- **High (0.2-0.3)**: Strong noise, very visible perturbations

## Adversarial Detection

Uses Laplacian variance to detect perturbations:
- **Variance < 100**: Likely adversarial
- **Variance 100-500**: Possibly adversarial
- **Variance > 500**: Likely original

Through this project, I learned about adversarial examples, defensive strategies, and understand model vulnerabilities
