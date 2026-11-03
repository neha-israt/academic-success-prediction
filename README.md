
# Academic Success Prediction

## Project Overview

This project uses machine learning to predict students' academic outcomes based on demographic, socioeconomic, and academic information.

The goal is to classify students into three possible outcomes:

- **Dropout** – Students who discontinue their studies.
- **Enrolled** – Students who remain enrolled.
- **Graduate** – Students who successfully complete their studies.

The project explores different machine learning models, evaluates their performance, and examines the factors that contribute most to predictions.

## Problem Statement

Student dropout is a significant challenge for educational institutions. Identifying students who may be at risk of dropping out can help institutions provide timely support and improve student retention.

This project investigates how machine learning can be used to predict academic outcomes using historical student data.

## Dataset

This project uses the **Predict Students' Dropout and Academic Success** dataset from the UCI Machine Learning Repository.

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- **Dataset size:** 4,424 student records
- **Features:** 36
- **Target variable:** `Target`
- **Classes:** Dropout, Enrolled, Graduate
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

The dataset contains information about students, including:

- Demographic information
- Previous academic performance
- Admission details
- First-semester academic performance
- Second-semester academic performance
- Socioeconomic factors

### Dataset Citation

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success*. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89.

The dataset is redistributed in this repository under its original CC BY 4.0 license. All rights belong to the original dataset authors.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Jupyter Notebook

## Project Workflow

### 1. Data Loading and Exploration

- Loaded the dataset and examined its structure.
- Checked data types, missing values, and duplicate records.
- Explored the distribution of academic outcomes.

### 2. Exploratory Data Analysis

- Analyzed relationships between student characteristics and academic outcomes.
- Visualized important patterns and distributions.

### 3. Data Preprocessing

- Separated features and target variable.
- Split the data into training and testing sets using stratification.
- Applied preprocessing techniques to prepare the features for machine learning.

### 4. Model Development

Four classification models were trained:

- Logistic Regression
- Random Forest
- XGBoost
- Multi-Layer Perceptron (MLP)

### 5. Model Evaluation

- Compared models using accuracy, precision, recall, and F1-score.
- Examined confusion matrices and classification reports.
- Evaluated performance across all three outcome classes.

### 6. Class Imbalance Handling

- Investigated the effect of class imbalance.
- Tested Logistic Regression with balanced class weights.
- Compared the trade-off between overall accuracy and minority-class recall.

### 7. Feature Importance

- Examined feature importance using Random Forest.
- Identified academic performance indicators as important predictive features.

### 8. Model Saving and Loading

- Saved trained models using Joblib.
- Saved the preprocessing pipeline.
- Tested loading the saved model and generating predictions.
- Verified prediction consistency between the original and loaded model.

## Model Comparison and Evaluation

Four machine learning algorithms were trained and evaluated to identify a suitable model for predicting student academic outcomes.

### Performance Comparison

| Model | Accuracy | Weighted F1-score | Macro F1-score |
|---|---:|---:|---:|
| Logistic Regression | 77.06% | 75.9% | 69.7% |
| Random Forest | 76.61% | 74.0% | 67.0% |
| XGBoost | 76.05% | 75.2% | 68.9% |
| MLP | 72.20% | 71.0% | 63.5% |

Logistic Regression achieved the highest overall accuracy and macro F1-score among the evaluated models.

However, accuracy alone does not provide a complete picture of model performance, particularly when the target classes are imbalanced.

### Class Imbalance Experiment

The `Enrolled` category was more difficult to predict than the other two classes.

To investigate this issue, a second Logistic Regression model was trained using:

```python
class_weight="balanced"
```

The performance of the original and balanced Logistic Regression models was compared.

| Metric | Original Logistic Regression | Balanced Logistic Regression |
|---|---:|---:|
| Accuracy | 77.06% | 72.50% |
| Weighted F1-score | 75.9% | 73.8% |
| Macro F1-score | 69.7% | 68.9% |
| Enrolled Precision | 55.9% | 42.0% |
| Enrolled Recall | 39.0% | 63.5% |
| Enrolled F1-score | 45.9% | 50.0% |

The balanced model improved recall for the Enrolled class from 39.0% to 63.5%.

However, this improvement came with a reduction in overall accuracy and precision.

### Model Selection

The original Logistic Regression model was retained as the primary model because it achieved the highest overall accuracy and macro F1-score among the evaluated models.

The balanced Logistic Regression model was also saved as an alternative because it demonstrated better recall for the Enrolled category.

This comparison highlights the importance of considering class-specific performance rather than selecting a model based on accuracy alone.

## Key Findings

- Logistic Regression achieved the highest overall test accuracy.
- Academic performance indicators were among the most influential predictive features.
- Second-semester approved course units and grades had particularly high feature importance.
- Class imbalance affected the model's ability to identify students in the Enrolled category.
- Balanced Logistic Regression improved Enrolled-class recall, although overall accuracy decreased.

## Prediction Timing and Feature Availability

Some highly influential features come from second-semester academic performance.

These features may not be available when institutions need to identify students at the beginning of their studies.

Therefore, the model's predictions should be interpreted in the context of the information available at prediction time.

A separate model using only early-available information would be more appropriate for early intervention.

## Limitations

- The model is trained on historical data and may not generalize equally well to other institutions or student populations.
- The Enrolled class remains more challenging to predict than the other two classes.
- Some features may only become available later in a student's academic journey.
- Feature importance indicates predictive contribution, not causation.

## Project Structure

```text
academic-success-prediction/
├── academic_success_prediction.ipynb
├── data.csv
├── logistic_regression_model.pkl
├── balanced_logistic_regression_model.pkl
├── preprocessor.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Navigate to the project directory

```bash
cd academic-success-prediction
```

### 3. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 5. Open the notebook

```bash
jupyter notebook academic_success_prediction.ipynb
```

### 6. Run the notebook

Run the notebook cells sequentially.

## Future Improvements

- Experiment with additional machine learning models and hyperparameter tuning.
- Improve prediction performance for the Enrolled class.
- Develop an early-warning model using only information available during the initial academic period.
- Apply explainable AI techniques to improve model interpretability.
- Build a simple web application to demonstrate student outcome predictions.

## Disclaimer

This project is intended for educational and research purposes. Predictions should not be used as the sole basis for academic decisions concerning individual students.
