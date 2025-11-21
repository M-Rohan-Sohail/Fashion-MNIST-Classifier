# 🛍️ Fashion MNIST Classifier – My First Deep Learning Project

## Overview
This is my **first step into Deep Learning**! In this project, I built a simple **image classification model** using TensorFlow to recognize items from the **Fashion MNIST dataset**.  

The project demonstrates how to:
- Load and preprocess image data  
- Normalize images for better model performance  
- Build a neural network with hidden layers to learn patterns  
- Train and evaluate a model for multi-class classification  

---

## Dataset
- **Dataset:** Fashion MNIST  
- **Number of Classes:** 10 (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)  
- **Image Size:** 28x28 grayscale images  
- **Normalization:** Image pixel values scaled between 0 and 1 for faster training  

---

## Model Architecture
- The model is a **fully connected neural network**  
- Includes a **hidden layer with 128 neurons** to learn complex patterns from the images  
- The **output layer predicts 10 classes** using probabilities  

---

## Training & Evaluation
- The model was trained for **100 epochs**.  
- **Optimizer used:** Adam, for efficient weight updates  
- **Loss function:** Sparse Categorical Crossentropy, suitable for integer class labels  
- **Metric:** Accuracy  

**Results after training:**
- **Training Accuracy:** 98.40%  
- **Test Accuracy:** 88.52%  

These results show that the model effectively learns from the training data and performs reasonably well on unseen test data.

---

## Key Learnings
- Understanding how to preprocess and normalize image data  
- Building a neural network for multi-class classification  
- Using activation functions like ReLU and Softmax  
- Evaluating model performance and interpreting results  

---

## Future Improvements
- Adding more hidden layers or dropout for better generalization  
- Implementing Convolutional Neural Networks (CNNs) for improved accuracy  
- Visualizing model predictions and misclassifications  

---

## Author
**Muhammad Rohan Sohail** – taking my first steps into **Deep Learning**!
