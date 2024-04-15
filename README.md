# Applied_Machine_Learning

## Predicting Academic Performance: Exploring Early Childhood Experiences and Family Environment Factors

This repository contains code for analyzing the Fragile Families and Child Wellbeing Study (FFCWS) dataset to investigate how early childhood experiences and family environment factors influence academic performance, as measured by grade point average (GPA) during adolescence.

### Dataset

The Fragile Families and Child Wellbeing Study (FFCWS) is a longitudinal study following nearly 5,000 children born between 1998 and 2000 in large U.S. cities. The dataset includes rich information on early childhood experiences, family dynamics, socioeconomic factors, and various child outcomes, including academic performance measures such as GPA.

### Project Overview

The main objectives of this project are:

1. **Data Preprocessing**: Clean and preprocess the FFCWS dataset, handling missing values, performing feature selection, and encoding categorical variables for machine learning models.

2. **Model Training**: Train and evaluate various machine learning models, including linear regression, logistic regression, decision trees, random forests, and gradient boosting, to predict GPA based on early childhood experiences and family environment factors.

3. **Hyperparameter Tuning**: Optimize the performance of the trained models by tuning their hyperparameters using techniques such as grid search and random search.

4. **Model Interpretation**: Interpret and explain the trained models using techniques like LIME (Local Interpretable Model-agnostic Explanations) and SHAP (SHapley Additive exPlanations) to understand the impact of different features on the predicted GPA.

## Requirements

To run the code in this repository, you'll need to have the following libraries installed:

- pandas
- numpy
- scikit-learn
- lime
- shap

You can install the required libraries using pip:

```
pip install pandas numpy scikit-learn lime shap
```

## Usage

1. Clone this repository to your local machine.
2. Download the FFCWS dataset from the Princeton University Fragile Families Study website (https://fragilefamilies.princeton.edu/) and place it in the `data/` directory.
3. Open the `analysis.ipynb` notebook in your preferred Jupyter environment.
4. Execute the cells in the notebook to preprocess the data, train and evaluate machine learning models, perform hyperparameter tuning, interpret the models, and analyze the effect of text length on classification performance.

## Contents

- `AML-Code-Final.ipynb`: A Jupyter Notebook containing the code for data preprocessing, model training, evaluation, interpretation, and text length analysis.
- `data/`: Directory where the FFCWS dataset should be placed.
- `utils.py`: Utility functions for data loading, visualization, and model interpretation.
- `requirements.txt`: List of required Python libraries and dependencies.
