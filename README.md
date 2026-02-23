
# Land–Water Boundary Detection using Custom CNN (From Scratch)

## Overview

This project presents a fully custom implementation of a Convolutional Neural Network (CNN) built from scratch (without using deep learning libraries such as TensorFlow or PyTorch) to detect land–water boundaries in satellite imagery.

The system focuses on separating smooth (water-like) and chaotic (land-like) regions using handcrafted convolutional feature extraction, patch-based analysis, and statistical decision logic. A Flask web application is integrated for interactive image upload and real-time boundary visualization.

This project was developed under academic constraints requiring:
- No deep learning frameworks
- No backpropagation
- Manual CNN implementation
- Emphasis on explainability and deterministic logic

---

## Problem Statement

Given a satellite image containing any type of water body (coastline, river, lake, muddy water, blue ocean, algae-rich water), the objective is to:

- Detect the land–water boundary
- Overlay the detected boundary on the original image
- Maintain robustness across varied terrains
- Avoid pretrained or black-box models

The focus is on boundary separation rather than full semantic segmentation.

---

## System Architecture

### 1. Preprocessing
- Image resizing to 256x256
- Grayscale conversion (for texture extraction)
- Normalization

### 2. Patch-Based Segmentation
- Sliding window (fixed patch size)
- Controlled stride
- Local texture analysis

### 3. Custom CNN (Feature Extractor)
- Manual convolution filters
- No training, no backpropagation
- Deterministic feature map generation
- Statistical feature extraction (mean, variance)

The CNN is used to measure the “chaos vs smoothness” level in each patch.

### 4. Boundary Formation
- Patch classification into smooth/chaotic regions
- Mask generation
- Contour extraction using OpenCV
- Red boundary overlay on original image

---

## Evaluation Strategy

Ground truth thick boundary masks were manually annotated using CVAT.

Instead of traditional IoU or pixel accuracy, a boundary coverage metric was used:

Boundary Score = (Predicted Boundary Pixels inside GT Region) / (Total Predicted Boundary Pixels)

This metric:
- Does not penalize tolerance
- Allows boundary thickness flexibility
- Evaluates alignment accuracy realistically

### Results

Evaluation on 10 diverse satellite test images:

Average Boundary Score: 0.64

Observations:
- Strong performance on muddy and algae-rich water bodies
- Moderate performance on clean blue water terrains
- Sensitive to patch size and stride configuration

---

## Deployment

A Flask-based web application was developed with:

- Image upload interface
- Real-time processing
- Boundary visualization
- Clean UI for demonstration

Run locally:

```bash
python app/app.py
```

---

## Project Structure

```
Land_Water_Separation_CustomCNN/
│
├── cnn/                  # Custom CNN implementation
├── segmentation/         # Patch-based segmentation logic
├── features/             # Statistical feature extraction
├── utils/                # Preprocessing pipeline
├── evaluation/           # Boundary metric computation
├── app/                  # Flask web application
├── data/                 # Test cases & ground truth masks
├── main.py               # Complete pipeline
└── README.md
```

---

## Key Highlights

- Custom CNN built entirely from scratch
- No deep learning frameworks used
- No backpropagation
- Fully explainable deterministic logic
- Real-world satellite imagery testing
- Integrated web deployment
- Custom boundary evaluation metric

---

## Skills Demonstrated

- Computer Vision fundamentals
- CNN architecture implementation
- Feature engineering
- Texture analysis
- Boundary detection logic
- Model evaluation design
- Flask web deployment
- Ground truth annotation workflow
- Academic project structuring

---

## Limitations

- Patch-grid dependency can cause boundary blockiness
- Blue water regions may require enhanced color modeling
- Performance depends on patch size and stride configuration

---

## Future Improvements

- Multi-scale patch analysis
- Adaptive stride near boundaries
- Edge-gradient assisted refinement
- Post-processing boundary smoothing
- Semi-supervised threshold calibration

---

## Conclusion

This project demonstrates that meaningful land–water boundary detection can be achieved using handcrafted CNN-based feature extraction without modern deep learning frameworks. The approach prioritizes interpretability, academic rigor, and engineering clarity.

It is suitable for:
- Academic submissions
- Resume demonstration
- GitHub portfolio projects
- Interview discussions on CNN fundamentals

