# Model Card: X-ray Image Classification using CNN with Bayesian Optimization
## Model Details
Model Details

Model Name: X-ray Image Classifier with CNN and Bayesian Optimization  
Model Version: 1.0  
Date of Model Creation: 01/07/2023  
Model Type: Convolutional Neural Network (CNN)  
Model Purpose: Medical image classification for X-ray diagnosis  
Model Architecture: LeNet5-based CNN with multiple convolutional and pooling layers  
Hyperparameter Optimization: Bayesian Optimization with BoTorch and trust region search  
  
## Model Overview
The X-ray Image Classifier with CNN and Bayesian Optimization is designed for accurate medical image classification for pneumonia in X-ray diagnosis.
The model leverages Convolutional Neural Networks (CNN) to analyze X-ray images and classify them into relevant medical conditions.
Bayesian Optimization with BoTorch and trust region search techniques are utilized to fine-tune hyperparameters, maximizing model performance while minimizing computational
resources and time.

## Intended Use
The model is intended to provide an example of how medical professionals can be assisted by AI technologies in diagnosing medical conditions from X-ray images more effectively and efficiently. 
It aims to identify true positive cases with high accuracy to minimize the risk of missing critical diagnoses. 
It is not intended to replace medical experts but rather to serve as a decision-support tool to aid in accurate and timely diagnoses.

## Training Data
- The model was trained on a labelled dataset of X-ray images containing both positive and negative instances of medical conditions.
- Experts thoroughly reviewed and preprocessed the dataset to ensure data quality and consistency.
- Data augmentation techniques were applied to increase the diversity of training samples and improve generalization.

## Performance Metrics
- The primary performance metric in this cycle is accuracy, however, Recall (Sensitivity) score should be the priority due to the nature of the project.
- Recall is prioritized over Precision as Type 2 error (false negatives) has higher consequences in a medical setting, where missing positive cases can lead to delayed treatment and compromised patient outcomes.
- Metrics such as Accuracy, Recall and F2-score are considered to provide a comprehensive assessment of the model's performance.

## Performance Results

- The trained model achieved a Recall score of approximately 0.98 on the validation dataset
- The high Recall score demonstrates the model's efficacy in accurately capturing a large proportion of positive cases, making it suitable for medical diagnosis tasks.

## Limitations
- As with any machine learning model, the X-ray Image Classifier has limitations. It may not generalize optimally to unseen data from different distributions or patient populations.
- The model's performance heavily depends on the quality and diversity of the training data. Inadequate or biased training data may lead to reduced performance and biased predictions.
- To mitigate the limitations, continuous monitoring and evaluation of model performance on new data are essential.
- Ongoing feedback from medical experts can aid in identifying areas for improvement and address any potential biases or limitations.

## Future Improvements
- In future model iterations, the goal should be achieving a Recall score of 1, ensuring that all positive instances are correctly identified without any false negatives.

