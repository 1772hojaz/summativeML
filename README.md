Lung Disease Prediction Project

Overview

This project aims to predict various types of lung diseases using chest X-ray (CXR) images. The goal is to build a reliable deep learning model to aid in early diagnosis, particularly in resource-limited settings. The model classifies 8 different lung diseases and 1 normal case, helping to identify conditions such as pneumonia, tuberculosis, and others early enough to improve patient outcomes.

Problem Statement

Lung diseases, especially those like tuberculosis and pneumonia, continue to be leading causes of morbidity and mortality, particularly in Zimbabwe. Early and accurate detection can significantly improve treatment outcomes and reduce healthcare costs. This project focuses on building a model capable of classifying various lung diseases using chest X-ray images.

Dataset

The dataset consists of chest X-ray images categorized into 9 classes (8 lung diseases and 1 normal case). The dataset is divided into training, validation, and testing sets. The images were sourced from publicly available datasets.

Findings

Instance

Model

Optimizer

Regularizer

Epochs

Early Stopping

Layers

Learning Rate

Accuracy

Loss

Precision

Recall

F1 Score

ROC AUC

1

CNN Model 1

None

None

5

No

9

0.001

0.74

0.51

0.166

0.66

0.163

0.51

2

CNN Model 2

Adam

None

5

Yes

9

0.001

0.73

0.50

0.15

0.15

0.15

0.50

3

CNN Model 3

RMSPROP

None

15

Yes

9

0.001

0.69

0.62

0.14

0.15

0.13

0.50

4

CNN Model 4

RMSPROP

L2

5

Yes

9

0.0001

0.7396

0.5267

0.139

0.139

0.137

0.498

5

LRegression

LBFGS

L2 (default)

10000

Yes

1

Managed by algorithm

0.5149

1.3681

0.8527

0.9073

0.8516



Summary

Best Model Among CNNs:

The CNN Model 1 achieved an accuracy of 74%, making it the best-performing CNN in this experiment. However, precision and recall remain low, indicating possible misclassification issues.

Overall Best Model:

CNN Model 4, which incorporated L2 regularization and early stopping, had slightly better overall generalization capability compared to other models.

Neural Networks vs. Traditional Machine Learning:

CNN models outperformed traditional logistic regression in accuracy. However, logistic regression had better precision and recall, which may be due to different optimization techniques.

Hyperparameters of Best Model (CNN Model 4):

Optimizer: RMSPROP

Regularization: L2

Epochs: 5

Early Stopping: Yes

Learning Rate: 0.0001

Installation & Usage

Requirements

Ensure you have the following dependencies installed:

pip install tensorflow keras numpy pandas matplotlib scikit-learn

Running the Best Saved Model

Clone the repository, navigate to the /lung_disease_prediction/saved_models/, and load the saved model to make predictions on new images.

import keras
from keras.models import load_model

# Load the CNN model
model = load_model('cnn_lung_disease_model.h5')

# Load your test data and make predictions
# test_data = load_your_test_data_function()
predictions = model.predict(test_data)

Conclusion

The CNN Model 4, with its regularization and early stopping, provided the best balance of performance and generalization. Future improvements could include:

 Increasing dataset size for better generalization.
 Experimenting with data augmentation (rotation, flipping, etc.).
 Using transfer learning with pre-trained models like ResNet or VGG16.
 Performing hyperparameter tuning with a more granular learning rate range.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements

We acknowledge the open datasets used in this project and the support of the AI research community.


