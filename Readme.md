# CGPA Predictor and Study Guidance System

A machine learning-powered Streamlit web app that predicts a student’s projected CGPA and provides practical study recommendations based on academic and lifestyle factors.

## Overview

This project was built to help students estimate their projected CGPA using indicators such as:
- Current CGPA
- Last SGPA
- Study hours
- Class attendance
- Continuous assessment score
- Past question practice
- Study group participation
- Sleep hours
- Work hours
- Extracurricular involvement
- Accommodation and academic level

The app combines:
- data cleaning and exploration
- model training and comparison
- CGPA prediction
- actionable study recommendations

It is designed as a student-focused decision-support tool rather than a strict academic forecasting system.

---

## Project Goal

The main goal of this project is to:
1. predict a student’s upcoming CGPA from available academic and behavioral data
2. understand which factors are strongly associated with academic performance
3. suggest small changes that can improve projected CGPA

---

## Features

- Interactive dashboard via Streamlit
- Student input form for manual CGPA prediction
- Model comparison dashboard
- Feature importance visualization
- Recommendation engine for study improvements
- CSV export of model comparison results
- Data exploration with distributions, correlations, and outliers

---

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- Streamlit
- Matplotlib

---

## Problem Framing

This is a supervised regression problem because the target variable is continuous:

- Target: `Projected_CGPA`
- Range: Typically between 0 and 5

The app learns patterns from historical student-like data and estimates the projected CGPA for a new student profile.

---

## Dataset

The project uses a synthetic student dataset stored in:

- `synthetic_cgpa_practice_data.csv`

The dataset includes fields such as:
- `Department`
- `Level`
- `Current_CGPA`
- `Last_SGPA`
- `Current_Credit_Load`
- `Carryovers`
- `Study_Hours_Per_Week`
- `Class_Attendance`
- `Continuous_Assessment_Score`
- `Past_Questions_Practice`
- `Study_Groups_Participation`
- `Sleep_Hours_Per_Night`
- `Extracurricular_Hours_Per_Week`
- `Work_Hustle`
- `Accommodation`
- `Projected_CGPA`

The training dataset is used internally to train and evaluate the models. The app does not let users browse or select student records, and the prediction form does not prefill values from the dataset.

---

## Data Preprocessing

Before training, the application:
- removes duplicate rows
- converts ordinal categorical values into numeric ranks
- converts accommodation status into binary form
- converts CGPA values to numeric and removes impossible values
- imputes missing numeric values using median values
- handles categorical data with one-hot encoding

This makes the data compatible with regression models.

---

## Model Training and Evaluation

The app compares two models:

### 1. Linear Regression
- interpretable
- strong baseline for structured tabular data
- suitable for clear coefficient-based insights

### 2. Random Forest Regressor
- captures nonlinear relationships
- handles interactions between factors more flexibly

The project uses:
- train-test split with 80/20 ratio
- 5-fold cross-validation
- RMSE, MAE, and R² as evaluation metrics

Model selection is based on the lowest cross-validated RMSE, then the best model is refit using the complete dataset.

---

## Recommendation Logic

The project does more than predict CGPA. It also suggests improvements.

For each actionable feature, the system:
- increases or decreases the value by one step
- evaluates the predicted CGPA change
- ranks the biggest positive gains
- presents the top suggestions to the user

Examples of actionable features include:
- study hours
- attendance
- assessment score
- past question practice
- study group participation
- work hustle
- extracurricular hours

This gives the user practical guidance such as:
- “Increase your weekly study hours”
- “Attend more classes”
- “Reduce time spent on work or business activities”

> These recommendations are model-based estimates from patterns in the data and should be interpreted as guidance, not guaranteed causal outcomes.

---

## Performance Metrics

The trained model comparison results are:

| Model | CV MAE | CV RMSE | CV R² | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|---:|
| Linear Regression | 0.153 | 0.188 | 0.934 | 0.140 | 0.185 | 0.945 |
| Random Forest | 0.166 | 0.210 | 0.913 | 0.168 | 0.205 | 0.932 |

### Interpretation
- The Linear Regression model performed best on this dataset.
- It achieved:
  - Test RMSE: 0.185
  - Test R²: 0.945
- This means the model explains about 94.5% of the variance in the test set and makes an average prediction error of roughly 0.18 CGPA points.

This is a strong result for a small educational dataset.

---

## App Structure

```text
CGPA ML PROJECT/
├── CGPA.py                  # Main Streamlit application
├── CGPA.ipynb              # Notebook version of the project
├── requirements.txt         # Python dependencies
├── synthetic_cgpa_practice_data.csv
├── model_comparison.csv    # Saved model evaluation results
├── eda_correlation.png     # Correlation visualisation
├── eda_distributions.png   # Distribution plots
├── feature_importance.png  # Feature importance visualisation
├── Readme.md               # Project documentation
└── AboutProject.txt        # Additional summary text