# Behind the Jab: Predicting H1N1 Vaccination Behavior

![H1N1 Vaccination Behavior](https://github.com/Day-hue/Phase-3-project/blob/main/Data/H1N1_vaccination_behavior.jpg)

## Overview
This project aims to predict  predict which individuals are most likely to refuse the vaccine,whether individuals will receive the H1N1 flu vaccine based on demographic, behavioral, and health-related factors. The goal is to build a machine learning model that can accurately classify individuals as likely or unlikely to get vaccinated. This can help public health organizations target vaccination campaigns more effectively.

The project uses a dataset from the National 2009 H1N1 Flu Survey (NHFS) conducted by the CDC (Centers for Disease Control and Prevention). The dataset includes responses from individuals about their vaccination status, health behaviors, and demographic information.

## Dataset
The dataset used in this project is available on DrivenData. It consists of two main files:

Training Data: Contains features and the target variable (H1N1 vaccination status).

Test Data: Contains features for which predictions need to be made.

## Features
The dataset includes the following types of features:

Demographic Information: Age, race, gender, education level, income, etc.

Health Behaviors: Doctor visits, health insurance status, etc.

Opinions and Beliefs: Perceived risk of H1N1, perceived effectiveness of the vaccine, etc.

## Target Variable
The target variable is binary:

1: The individual received the H1N1 vaccine.

0: The individual did not receive the H1N1 vaccine.

## Methodology
The project follows a standard machine learning workflow:

### Data Preprocessing:

Handling missing values (imputation).

Encoding categorical variables.

Scaling numerical features.

Exploratory Data Analysis (EDA):

Visualizing distributions of features.

Analyzing correlations between features and the target variable.

Feature Engineering:

Creating new features based on domain knowledge.

Selecting relevant features using techniques like feature importance or correlation analysis.

### Modeling:

Training machine learning models (e.g.  Random Forest and XBGBOOST).

Hyperparameter tuning using Grid Search or Random Search.

### Evaluation:

Evaluating models using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

Cross-validation to ensure model robustness.


## Requirements
To run the code, you need the following Python libraries:

NumPy

Pandas

Scikit-learn

Matplotlib

Seaborn

XGBoostClassifier 

RandomForestClassifier

Plotly

For more info please check the notebook  
