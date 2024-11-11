# Fracture Detection Project

This project is designed to detect fractures in X-ray images, differentiating between fractured and non-fractured cases using convolutional neural networks (CNNs) and transfer learning models (VGG19 and DenseNet). An ensemble approach using hard voting combines predictions from multiple models for more accurate results.


## Requirements
- Python 3.x
- TensorFlow
- Matplotlib
- NumPy

## Data Preprocessing
1. **Load Dataset**: Load images from the `FracAtlas/images` directory and resize them to 224x224 pixels.
2. **Normalize Images**: Scale pixel values to [0, 1] for better model performance.
3. **Remove Corrupt Images**: A function `find_and_delete_corrupt_images` checks and removes any unreadable or corrupted images.
4. **Dataset Split**: Split the dataset into training, validation, and test sets (70%, 20%, and 10%, respectively).

## Model Training
Three models are used:
1. **CNN Model**: A simple custom CNN with convolutional, pooling, and fully connected layers.
2. **VGG19 Model**: A pre-trained VGG19 model (ImageNet weights) with a custom classification head.
3. **DenseNet121 Model**: A pre-trained DenseNet121 model with L2-regularized layers to prevent overfitting.

### Training Steps
- **Compile and Train** each model on the training set, with early stopping on the validation set to avoid overfitting.

## Ensemble Model - Hard Voting
An ensemble approach with hard voting is implemented, combining predictions from the CNN, VGG19, and DenseNet121 models. The function `hard_voting_ensemble` outputs the majority vote among the models' predictions.

## Prediction
To make a prediction on a new image:
1. **Preprocess Image**: Load and resize to 224x224, normalize, and expand dimensions for model compatibility.
2. **Run Prediction**: Use the `hard_voting_ensemble` function to get the final class prediction based on combined model outputs.