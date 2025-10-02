***

# Titanic Survival Predictor - Classification Algorithm Implementation

This repository contains a hands-on notebook where I implement and compare various supervised machine learning classification algorithms using the Titanic dataset from seaborn. The aim is to demonstrate model building, evaluation, and comparison on a classic dataset using Python and scikit-learn.

***

## Dataset

- The notebook works with the Titanic dataset, which is directly loaded with sns.load_dataset('titanic').
- The dataset includes features such as passenger class, sex, age, fare, and whether the passenger survived.

***

## Implemented Algorithms

I explored and evaluated the following classification models:
- *Logistic Regression*
- *K-Nearest Neighbors (KNN)*
- *Gaussian Naive Bayes*
- *Decision Tree Classifier*
- *Support Vector Machine (SVM, RBF kernel)*

Each model is built using scikit-learn and evaluated for precision, recall, f1-score, accuracy, and confusion matrix.

***

## Results

After preprocessing and splitting the data, I trained and tested each model. Here are the performance metrics on the test set:

| Model                | Accuracy | Precision (0/1) | Recall (0/1) | F1-score (0/1) |
|----------------------|----------|-----------------|--------------|----------------|
| Logistic Regression  | 0.81     | 0.87 / 0.74     | 0.83 / 0.80  | 0.85 / 0.77    |
| KNN (k=5)            | 0.77     | 0.84 / 0.71     | 0.81 / 0.75  | 0.82 / 0.73    |
| Gaussian Naive Bayes | 0.81     | 0.87 / 0.73     | 0.81 / 0.81  | 0.84 / 0.77    |
| Decision Tree        | 0.79     | 0.79 / 0.78     | 0.85 / 0.70  | 0.82 / 0.74    |
| SVM (RBF kernel)     | 0.82     | 0.85 / 0.77     | 0.85 / 0.77  | 0.85 / 0.77    |

- *SVM and Logistic Regression provided the highest accuracy and balanced metrics in this setup.*
- All models used standard evaluation via test set accuracy, macro/weighted averages, and class-specific results.

***

## Structure

- *Titanic_survival_predictor(Using_Classification_Algos).ipynb*: Main analysis notebook with detailed code and outputs.
- Step-by-step model fitting, prediction, and performance evaluation for each algorithm.
- Includes data preprocessing, train-test splitting, and result interpretation.

***

## Requirements

- Python 3.x
- pandas, numpy
- seaborn
- matplotlib
- scikit-learn

Install these with:

pip install pandas numpy seaborn matplotlib scikit-learn


***

Feel free to fork or contribute ideas if you'd like to extend the analysis!
