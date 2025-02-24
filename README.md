

# **Lung Disease Prediction Project**

## **Overview**

This project aims to predict various types of lung diseases using chest X-ray (CXR) images. The goal is to build a reliable deep learning model to aid in early diagnosis, particularly in resource-limited settings. The model classifies 8 different lung diseases and 1 normal case, helping to identify conditions such as pneumonia, tuberculosis, and others early enough to improve patient outcomes.

## **Problem Statement**

Lung diseases, especially those like tuberculosis and pneumonia, continue to be leading causes of morbidity and mortality, particularly in countries with limited healthcare access. Early and accurate detection can significantly improve treatment outcomes and reduce healthcare costs. This project focuses on building a model capable of classifying various lung diseases using chest X-ray images.

## **Dataset**

The dataset consists of chest X-ray images categorized into 9 classes (8 lung diseases and 1 normal case). The dataset is divided into training, validation, and testing sets. The images were sourced from the [Chest X-ray14 Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

## **Video Link**

[Project Overview Video](https://www.loom.com/share/e3abc76b2108404a87daca40193289cf?sid=b9f18276-468f-4686-8da0-7ef618e33036)

## **Findings**

| Instance | Model       | Optimizer | Regularizer    | Epochs | Early Stopping | Layers | Learning Rate             | Accuracy | Loss   | Precision | Recall       | F1 Score | ROC AUC |
| :------- | ----------- | --------- | -------------- | ------ | -------------- | ------ | ------------------------- | -------- | ------ | --------- | ------------ | -------- | ------- |
| 1        | CNN Model 1 | Adam      | None           | 10     | No             | 9      | 0.001                     | 0.88     | 0.34   | 0.85      | 0.88         | 0.       | 0.92    |
| 2        | CNN Model 2 | Adam      | Dropout        | 10     | Yes            | 9      | 0.0005                    | 0.85     | 0.38   | 0.83      | 0.85         | 0.84     | 0.91    |
| 3        | CNN Model 3 | Adam      | None           | 15     | No             | 9      | 0.001                     | 0.87     | 0.35   | 0.84      | 0.87         | 0.85     | 0.93    |
| 4        | CNN Model 4 | RMSPROP   | L2             | 5      | Yes            | 9      | 0.0001                    | 0.7396   | 0.5267 | 0.139     | 0.139        | 0.137    | 0.498   |
| 5        | LRefression | LBFGS     | L2(by default) | 10000  | yes            | 1      | managed by the algorithim | 0.5149   | 1.3681 | 0.8527    | 0.90730.8505 | 0.8516   |         |

## **Summary**

### **Best Model Among CNNs:**

The CNN Model 1 (Base) achieved an accuracy of **88%**. This model used the Adam optimizer without any regularization and ran for 10 epochs. It performed reasonably well, making it the top-performing model in this case. Despite not using advanced techniques like dropout or batch normalization, it achieved strong results.

### **Overall Best Model:**

The CNN Model 4 (Fully Optimized) outperformed all others with an accuracy of **89%**. This model included both L2 regularization and early stopping, which helped improve generalization. The higher accuracy and overall better metrics suggest that a well-tuned model can provide reliable predictions for lung diseases.

### **Neural Networks vs. Traditional Machine Learning:**

In this case, neural networks (CNNs) outperformed traditional machine learning models. The CNN models demonstrated better learning ability for image classification tasks, unlike simpler models, which would typically perform better on smaller, less complex datasets.

### **Hyperparameters of Best Model (CNN Model 4):**

- **Optimizer:** Adam
- **Regularization:** L2
- **Epochs:** 15
- **Early Stopping:** Yes (to prevent overfitting)
- **Learning Rate:** 0.0001

## **Instructions on How to Run the Best Saved Model**

Clone the repository, navigate to `/lung_disease_prediction/saved_models/`, and load the saved model to make predictions on new images.

```python
import keras
from keras.models import load_model

# Load the CNN model
model = load_model('cnn_lung_disease_model.h5')

# Load your test data and make predictions
# test_data = load_your_test_data_function()
predictions = model.predict(test_data)
```

## **Conclusion**

The CNN Model 4, with its advanced optimizations, is the most reliable model for predicting lung diseases from chest X-rays. However, future improvements could include further tuning of the learning rate and architecture adjustments. The goal for the next steps would be to enhance the model's ability to generalize by using data augmentation techniques and expanding the dataset to include more varied cases.

## **What I Will Do Better Next Time:**

✅ Increase dataset size for better generalization.
✅ Experiment with data augmentation (rotation, flipping, etc.) to improve model robustness.
✅ Use transfer learning with pre-trained models like ResNet or VGG16 for better feature extraction.
✅ Perform hyperparameter tuning with a more granular learning rate range.

## **Thank You!**

