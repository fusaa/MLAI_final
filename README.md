# X-ray Image Classification using CNN with Bayesian Optimization

This is the final project for Professional Certificate in Machine Learning & Artificial Intelligence programme offered by the Imperial College Business School. The goal for this project is to apply Bayesian Optimization to find the best hyperparameters for a neural network model with CNN. 

The chosen dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). The dataset comprehends lung X-ray images from children, and the goal is to train a classifier that is able to diagnose patients for pneumonia. 

> Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification”, Mendeley Data, V2, doi: 10.17632/rscbjbr9sj.2
---
# Data
The dataset has high-resolution images, and it has been pre-cleaned and labelled by specialists. It comprehends 5.856 image files with a size of 1.15GB.  

![image](https://github.com/fusaa/MLAI_final/assets/66756007/4982298c-8eea-461a-a4e7-25004e0d9acf)

--
# The Model

This model was inspired by a LeNet-5 Network. It is a convolutional neural network (CNN) architecture developed by Yann LeCun in 1998 for handwritten digit recognition. It consists of seven layers, including two convolutional layers, two pooling layers, and two fully connected layers. The first convolutional layer extracts features using a 5x5 filter, followed by a pooling layer. The second convolutional layer extracts more complex features using a 5x5 filter, followed by another pooling layer. The extracted features are then flattened and fed into two fully connected layers, with the final layer producing the classification output. LeNet-5 played a significant role in popularizing CNNs and laid the foundation for modern deep-learning applications.


---

![image](https://github.com/fusaa/MLAI_final/assets/66756007/96e486ad-9ed5-42ab-89d3-35249e5fdc16)
  
![image](https://github.com/fusaa/MLAI_final/assets/66756007/91017d9d-155f-41a7-8398-24a7f227d3df)  
[Source](https://www.datasciencecentral.com)
  
The traditional LeNet-5 architecture was used as a prototype for a versatile model that will be optimised.  

The model has been rewritten to allow for different hyperparameter definitions:

    Number of ouputs of each convolutional layer
    Kernel size used in each convolutional layer
    Filter Size and Stride Step for MaxPool layer

That makes six parameters optimised in this project so far out of eight parameters(including the optimizer learning rate and momentum).
In our case, the input layer consists of resized images to 224x224 pixels.

---
# Method
First, a set of hyperparameter data points was generated using random numbers, which later served as the initial data for the Bayesian Optimizer.
It is crucial to highlight that Trust Region Search was employed using BoTorch.

After the data points were created, the model was queried to obtain the results for these points.

The results were then stored in various lists, and subsequently, a pandas dataframe was generated, incorporating both the hyperparameters and the corresponding performance results.

Bayesian Optimization was then conducted to search for the best hyperparameters to achieve the highest accuracy possible.

The script executing the BoTorch was run several times, and results stored in a pickle file.

After a sufficient number of good models had been obtained, the possibility of selecting a model to proceed with emerged.
The model was subjected to testing across various metrics, such as Recall and F2Score. 
Additionally, a confusion matrix was constructed both for validation and testing - so far unseen data - that had been previously separated.

# Results

Best model settings:  
![image](https://github.com/fusaa/MLAI_final/assets/66756007/e8128ee6-a0f4-4a85-bf15-6f11006d6911)

Confusion Matrix:  
![image](https://github.com/fusaa/MLAI_final/assets/66756007/01322a52-ebc7-4553-8278-926454dee51a)

In this kind of problem the type of error plays a role in the usefulness of the model, and how it could be applied in a real world task. So it is necessary to consider the two types of errors:

> **False Positive (Type 1 Error):** You predicted positive and it’s false. You predicted that animal is a cat but it actually is not (it’s a dog).  
> **False Negative (Type 2 Error):** You predicted negative and it’s false. You predicted that animal is not a cat but it actually is.

In this case of a medical environment, having to choose the kind of error that the model can be optimized for, what would be the proper choice? If you classify a patient as ```normal```, when in fact, he has pneumonia, he won't be able to get appropriate treatment. However, suppose you classify a health patient as positive for ```pneumonia```. In that case, he will get through to a doctor to do further investigations on his particular case and the doctor will conclude if the patient has pneumonia or not.  

So, in this case, the proper course of action would be to minimize False Negatives(Type 2 error). Telling a patient he does not have a condition when he has will prevent further medical investigation into his case, and can lead to patient not getting proper treatment.

In this context, Recall is an important metric, and it is defined as the ratio of the total number of correctly classified positive classes divided by the total number of positive classes.
$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$


![image](https://github.com/fusaa/MLAI_final/assets/66756007/0273f59a-842a-4d6e-88d8-f76e550cd628)

As we can see, for the Recall metric(629 / (629+13) ), we correctly identified almost 98% of patients that were sick as sick, leaving roughly 2% misdiagnosed. In this context, in a review cycle for this work, the Recall performance should be prioritized to achieve 100%, even though it might mean getting more errors type 1.  

On the other hand, F-Score is a measurement that combines both Precision and Recall; by setting F to 2, we will give more importance to Recall.

$$F_\beta = \frac{(1 + \beta^2) \cdot \text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}$$


>*Fβ-score Formula Symbol*  
>A factor indicating how much more important recall is than precision. For example, if we consider recall to be twice as important as precision, we can set β to 2. The standard F-score is equivalent to setting β to one.[1](https://deepai.org/machine-learning-glossary-and-terms/f-score)  
  
We can check that the F2Score is compatible with the Recall Score.
The results obtained were similar when giving the recall score twice the importance over the Precision metrics.
This ensures that we do not obtain a seemingly good Recall measurement merely because the model is classifying a vast majority of instances as positive,
and consequently, achieving a high Recall score. Consequently, the F2Score aids in verifying that the high Recall observed is not solely a result of indiscriminate 
positive classifications, but rather a genuine ability of the model to make accurate predictions, striking an essential balance between Recall and Precision.  

# Conclusion
Bayesian optimization has been widely used in achieving exceptional results. By combining Bayesian optimization with BoTorch Trust Region Search, we enhanced the performance of our LeNet5-based CNN model.

Bayesian Optimization is a powerful technique that excels in scenarios where the objective function is costly to evaluate, as is often the case in complex machine learning models. It leverages a probabilistic model to intelligently explore the hyperparameter space, making informed decisions about which configurations to try next. This capability becomes crucial when training deep learning models, which typically require significant computational resources and time.

In this project, Bayesian optimization played a vital role in fine-tuning hyperparameters to maximize the optimization metric. This should be considered an initial work for reviewing. The focus for the following review should be minimizing Type 2 errors. This will reduce the risk of missing positive cases during diagnoses, ultimately improving patient outcomes and healthcare efficiency.

Botorch provided a flexible and modular platform for implementing Bayesian optimization algorithms. At the same time, trust region search efficiently explored the search space yielding better results with less computational power.

The Recall score of approximately **0.98** achieved in our project demonstrates the efficacy of Bayesian optimization in optimizing the model's performance for medical diagnosis tasks. However, there is always room for improvement. Exploring additional hyperparameter configurations and incorporating more sophisticated models into the optimization pipeline may yield even better results in future iterations.

Bayesian optimization was a critical component of this project, playing a key role in fine-tuning hyperparameters and achieving exceptional metrics. Its ability to optimize complex models in computationally efficient ways proved instrumental in creating a robust and reliable X-ray image classification system for medical diagnosis. Leveraging the power of Bayesian optimization, the model will continue to evolve, ultimately leading to better metrics.

