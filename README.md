# Stanford Dogs Breed Classification

Deep learning model for dog breed classification based on the Stanford Dogs Dataset. Uses EfficientNetV2S architecture with pre-trained weights.

## Features
- Transfer learning with EfficientNetV2S
- Data augmentation for better training
- Two-stage training: frozen layers first, then fine-tuning
- GPU support via TensorFlow

## Requirements
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib

## Model Architecture
- Base model: EfficientNetV2S (pre-trained on ImageNet)
- Additional layers: GlobalAveragePooling2D, BatchNormalization, Dropout
- Output layer: Dense with softmax activation

## Performance Metrics
- Training and validation accuracy
- Learning curves for model evaluation

## Dataset
The Stanford Dogs Dataset contains images of 120 breeds of dogs. The model is trained to classify these breeds with high accuracy using transfer learning techniques.
