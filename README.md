# Food Freshness Classification using Deep Learning

This project is a submission for the "Selected Topics in Computer Science (CoSc4132)" course at Debre Berhan University. It implements a Deep Learning model using Convolutional Neural Networks (CNN) and Transfer Learning to classify the freshness of fruits and vegetables.

## 1. Project Overview

The main objective of this project is to build an efficient image classification model that can accurately distinguish between fresh and rotten food items. By leveraging the power of pre-trained models, we aim to achieve high accuracy with minimal training time and computational resources. The model is trained on a public dataset and can classify images of apples, bananas, and oranges into "fresh" or "rotten" categories.

**Project Type:** CNN Transfer Learning (Image Domain)

## 2. Dataset

The dataset used for this project is the "Food Freshness Dataset" which contains images of fresh and rotten fruits.

- **Source:** [የ Kaggle ወይም Hugging Face ሊንኩን እዚህ አስገባ]
- **Total Images:** 13,589 images
- **Classes (6):** `freshapples`, `freshbananas`, `freshoranges`, `rottenapples`, `rottenbananas`, `rottenoranges`
- **Data Split:**
  - Training Set: 10901 images
  - Validation Set: 2698 images

## 3. Methodology

### 3.1. Model Architecture
We employed Transfer Learning using the **MobileNetV2** architecture, pre-trained on the ImageNet dataset. The architecture consists of the MobileNetV2 base model and a custom classification head composed of a `GlobalAveragePooling2D` layer and a `Dense` layer with `softmax` activation.

### 3.2. Data Augmentation
To prevent overfitting and make the model more robust, we applied data augmentation techniques to the training images, including rescaling, rotation, shifting, shear, zoom, and horizontal flipping.

## 4. Results and Evaluation

The model was trained for **4 epochs** and evaluated on the validation set.

### 4.1. Training Performance
The training and validation accuracy/loss curves show effective learning without significant overfitting.

![Training and Validation Performance](accuracy_loss_plot.png)

### 4.2. Classification Report
The classification report provides a detailed breakdown of the model's performance.

| Class          | Precision | Recall | F1-Score |
|----------------|-----------|--------|----------|
| freshapples    | 0.98      | 0.99   | 0.98     |
| freshbananas   | 0.99      | 0.97   | 0.98     |
| freshoranges   | 0.99      | 0.99   | 0.99     |
| rottenapples   | 0.99      | 0.97   | 0.98     |
| rottenbananas  | 0.98      | 0.99   | 0.98     |
| rottenoranges  | 0.98      | 0.98   | 0.98     |
| **Accuracy**   |           |        | **0.98** |

### 4.3. Confusion Matrix
The confusion matrix below visualizes the model's performance, showing very few misclassifications.

![Confusion Matrix](confusion_matrix.png)

## 5. Model Prediction Examples

Here are some examples of the model's predictions on new images.

#### Correct Predictions
![Correct Predictions](correct_predictions.png)

#### Incorrect Predictions
![Incorrect Predictions](incorrect_predictions.png)

## 6. Model Limitations (Out-of-Distribution Data)
A key limitation is the model's inability to handle images outside its training distribution. For instance, when given an image of a cat, it incorrectly predicts "freshoranges".

![Cat Prediction](cat_prediction.png)

## 7. How to Run the Project

1. **Clone the repository:** `git clone [Your Repository URL]`
2. **Open the Notebook:** Open `Food_Freshness_Classification.ipynb` in Google Colab.
3. **Run the cells:** Execute the cells in order to see the results.
