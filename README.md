# MNIST Classification
Deep Learning project using PyTorch to classify handwritten digit

# Handwritten Digit Recognition using Deep Learning

This project demonstrates how to classify handwritten digits (0–9) from the MNIST dataset using a deep learning model built with PyTorch.

## Objective
Build a neural network that can accurately predict handwritten digits from the MNIST dataset using basic deep learning techniques.

## Project Structure
Number Classification/
- data # MNIST Dataset
- Training_Eval.py # Model, Training loop and evaluation
- run.py # Runs the model
- model.pth # Saved model
- Sample # Sample output images (optional)
- report.docx # Internship report
- requirements.txt # Required Python packages
- README.md # Project overview

## Model Overview
Framework: PyTorch

Dataset: MNIST (28x28 grayscale images of digits 0–9)

Architecture: Simple Feedforward Neural Network

Accuracy: ~96% on test data

## How to Run
1. Train the Model

Training_Eval.py

This will:

Train the model on the MNIST dataset

Save the model as model.pth

2. Predict with Your Own Image
Ensure your image:

Is a white digit on a black background

Is resized to 28x28

Save it on Samples

Is in grayscale

Then run:

run.py 

## Requirements

Install all dependencies with:

pip install -r requirements.txt

## Author
- Manoj Kumar S
- Thiagarajar College of Engineering
- Internship Guide: Shankar Nivas Manickam
- Company: Sailors Inc.
- Date: 02.08.2025

## Sample Results
### Sample Prediction

![MNIST Prediction](Number%20Classification/Sample/sample1.png)

**Predicted Digit:** 7  
**Confidence per digit:**  
- 0 → 0.00%  
- 1 → 0.00%  
- 2 → 0.03%  
- 3 → 0.01%  
- 4 → 0.00%  
- 5 → 0.00%  
- 6 → 0.00%  
- 7 → 99.72%  
- 8 → 0.24%  
- 9 → 0.00%
 
## Notes

This repo is for educational/internship purposes.

Dataset is auto-downloaded by PyTorch's torchvision.datasets.

