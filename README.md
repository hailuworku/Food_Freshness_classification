            Food Freshness Classification using Deep Learning
This project is a submission for the Selected Topics in Computer Science (CoSc4132) course at Debre Berhan University. It implements a Deep Learning model using Convolutional Neural Networks (CNN) and Transfer Learning to classify the freshness of fruits and vegetables.
1. Project Overview
The main objective of this project is to build an efficient image classification model that can accurately distinguish between fresh and rotten food items. By leveraging the power of pre-trained models, we aim to achieve high accuracy with minimal training time and computational resources. The model is trained on a public dataset and can classify images of apples, bananas, and oranges into "fresh" or "rotten" categories.
Project Type: CNN Transfer Learning (Image Domain)
2. Dataset
The dataset used for this project is the "Food Freshness Dataset" which contains images of fresh and rotten fruits.
Source: [Provide the Kaggle or Hugging Face link here]
Total Images: [e.g., 13,589 images]
Classes (6):
freshapples
freshbananas
freshoranges
rottenapples
rottenbananas
rottenoranges
Data Split:
Training Set: 10901 images
Validation Set: 2698 images
3. Methodology
3.1. Model Architecture
We employed Transfer Learning using the MobileNetV2 architecture, pre-trained on the ImageNet dataset. Transfer learning allows us to use the knowledge (feature extraction capabilities) learned from a massive dataset and apply it to our specific task.
The architecture consists of:
Base Model: The convolutional base of MobileNetV2, with its weights frozen initially to retain the learned features.
Custom Head: We added a custom classification head on top of the base model:
A GlobalAveragePooling2D layer to reduce the feature map dimensions.
A Dense layer with a softmax activation function for multi-class classification.
3.2. Data Augmentation
To prevent overfitting and make the model more robust, we applied data augmentation techniques to the training images using ImageDataGenerator:
Rescaling (Normalization)
Rotation
Width and Height Shift
Shear and Zoom
Horizontal Flip
4. Results and Evaluation
The model was trained for [e.g., 4 epochs] and evaluated on the validation set.
4.1. Training Performance
The training and validation accuracy and loss curves show that the model learned effectively without significant overfitting.

  # image or screenshoot not upload
   
4.2. Classification Report
The classification report provides a detailed breakdown of the model's performance for each class, including precision, recall, and F1-score.
Class	Precision	Recall	F1-Score	Support
freshapples	0.98	0.99	0.98	394
freshbananas	0.99	0.97	0.98	383
freshoranges	0.99	0.99	0.99	385
rottenapples	0.99	0.97	0.98	600
rottenbananas	0.98	0.99	0.98	530
rottenoranges	0.98	0.98	0.98	406
Accuracy			0.98	2698
Macro Avg	0.98	0.98	0.98	2698
Weighted Avg	0.98	0.98	0.98	2698
(Note: Replace the table above with your actual classification report results.)
4.3. Confusion Matrix
The confusion matrix below visualizes the model's performance. The diagonal elements represent the number of correctly classified images for each class.
  # image or screenshoot not upload
  
The matrix shows that the model performs exceptionally well, with very few misclassifications between classes.
6. Model Prediction Examples
Here are some examples of the model's predictions on new images.
Correct Predictions
  # image or screenshoot not upload#
  
Incorrect Predictions
  # image or screenshoot not upload
  

7. Model Limitations (Out-of-Distribution Data)
A key limitation of this model is its inability to handle images that are outside its training distribution (i.e., non-food images). When presented with an image of a cat, the model incorrectly classifies it as "freshoranges," likely due to color similarity.
  # image or screenshoot not upload
