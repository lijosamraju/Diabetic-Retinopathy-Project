# Diabetic Retinopathy Detection using Deep Learning

## Project Overview

This project implements a deep learning solution to detect Diabetic Retinopathy (DR) from retinal fundus images. The goal is to create a reliable screening tool that can classify images as either "Healthy" or "DR Present." This has a significant real-world impact by helping to automate the screening process and potentially save patients' sight through early detection.

The model is a Convolutional Neural Network (CNN) built with Python and TensorFlow/Keras.

## Dataset

The project uses the **APTOS 2019 Blindness Detection** dataset from Kaggle.
- The original dataset contains over 3,600 images with 5 diagnostic levels (0-4).
- For this project, the problem was simplified to a binary classification (0 for No DR, 1 for DR Present).
- A balanced subset of 1,000 images (500 healthy, 500 with DR) was used for training and validation to ensure the model learned effectively without bias.

## Technologies Used
- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- Matplotlib & Seaborn
- Jupyter Notebook

## Results

After training for 15 epochs, the model achieved a **Validation Accuracy of 93%**.

### Performance Metrics

The model's performance on the unseen validation set is summarized below:

**Confusion Matrix:**
*(Screenshot your confusion matrix and place it here)*

**Classification Report:**
| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| No DR (0) | 0.88 | 0.98 | 0.93 |
| DR Present (1) | 0.98 | 0.87 | 0.92 |

### Key Insights
- The model shows a very high **precision (98%)** for detecting DR, meaning it generates very few false alarms.
- The model has a strong **recall (87%)** for DR cases, though this is the primary area for future improvement to minimize missed diagnoses (false negatives).

## How to Run

1. Clone the repository.
2. Download the `train_images.zip` and `train.csv` from the [Kaggle competition page](https://www.kaggle.com/c/aptos2019-blindness-detection/data).
3. Unzip the images into a `train_images` folder in the project directory.
4. Ensure you have the required libraries installed (`pip install tensorflow scikit-learn pandas matplotlib seaborn tqdm`).
5. Open and run the `.ipynb` file in Jupyter Notebook.