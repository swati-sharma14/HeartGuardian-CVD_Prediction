# HeartGuardian (Cardiovascular Disease Prediction using Framingham Dataset)

## Project Description
🩺 This project focuses on predicting cardiovascular diseases using traditional machine learning models and incorporating concepts from survival analysis. The Framingham dataset serves as the foundation for our predictive models.

## Motivation
🔍 Detecting and predicting cardiovascular diseases early can significantly improve patient outcomes. This project aims to explore the predictive capabilities of traditional ML models and survival analysis techniques to enhance risk stratification.

## Dataset Identification
📊 The Framingham dataset, a comprehensive collection of cardiovascular health data, is utilized for training and evaluating the predictive models. The dataset includes various features such as age, cholesterol levels, blood pressure, and lifestyle factors. This section involves identifying the dataset, exploring its characteristics through data visualization, and determining the optimal pre-processing steps required for model training.

## Risk Stratification & Survival Analysis
📈 We employ risk stratification techniques and survival analysis concepts to understand the progression of cardiovascular diseases over time. This involves identifying high-risk individuals and predicting their likelihood of developing the disease.

## Literature Review
📚 A comprehensive literature review is conducted to understand the existing approaches to cardiovascular disease prediction. This helps us incorporate best practices and state-of-the-art methodologies into our models. We compare traditional machine learning models with survival analysis models and specifically implement the Cox Proportional Hazards (CoxPH) model for its effectiveness in handling time-to-event data.

## Code - Kaplan Meier Estimator, Mortality Prediction Package
👩‍💻 The code includes implementations of the Kaplan Meier estimator for survival analysis and a mortality prediction package that facilitates risk assessment based on various input features. It also includes the implementation of the CoxPH model and the pre-processing steps required to prepare the dataset for survival analysis.

## Code - Random Survival Forest
👩‍💻 The code includes the implementation of the Random Survival Forest model, showcasing an alternative approach to survival analysis for predicting cardiovascular disease outcomes.

## Results and Analysis
📊 After training and evaluating the models, we present the results along with in-depth analysis. This includes model performance metrics, feature importance, and insights gained from survival analysis.
The results from the initial training process were:
Mean C-index = 0.7585
Mean Ctd-index = 0.7303

## Presentation Slides
🔗 The presentation slides provide a concise overview of the project, highlighting the motivation, dataset details, methodology, results, and implications. They serve as a valuable resource for communicating key findings.

Feel free to explore the code, datasets, and presentation slides provided in this repository to gain a deeper understanding of our cardiovascular disease prediction project.
