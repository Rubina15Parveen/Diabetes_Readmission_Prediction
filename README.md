# END TO END ML PROJECT IMPLEMENTATION

## Objective
This project applies machine learning to predict whether diabetic patients will be readmitted within 30 days of a hospital encounter. Early identification of high-risk patients enables better care planning and reduces healthcare costs.

I developed a binary classification pipeline (readmitted <30 days />30 days vs. NO) using the Diabetes 130-US Hospitals dataset. The project also includes a Flask-based GUI that allows users to input patient features and get real-time readmission risk predictions.

## Business Context

Hospital readmissions are a major cost driver and impact both patient outcomes and insurer risk. By flagging high-risk patients, an insurance company can:

- Anticipate future claims

- Offer preventive care programs

- Reduce avoidable hospitalizations

- Improve resource allocation

## Tech Stack

* Languages/Tools: Python, Pandas, NumPy, Scikit-learn, XGBoost, CatBoost

* Visualization: Matplotlib, Seaborn

* Deployment: Flask, HTML/CSS for GUI

* Other: SQLite for persistence, Joblib for model saving

## Approach

1. EDA: demographic analysis, missing value handling, ICD-9 diagnosis grouping.

2. Preprocessing: imputation, categorical encoding, train-validation split.

3. Models Tested:

    - Logistic Regression (baseline)

    - Random Forest

    - CatBoost

    - XGBoost (final choice)

4. Evaluation: AUC, Precision, Recall, F1-score, Confusion Matrix.

5. Deployment: Flask GUI where users can enter patient details and view predicted readmission risk.


## Results

- Logistic Regression: AUC ~0.69, balanced but weak.

- Random Forest: AUC ~0.70, balanced precision/recall (~0.63).

- CatBoost: AUC ~0.71, Recall ~0.87 (higher sensitivity, more false positives).

- XGBoost (Final Model): AUC ~0.71, Recall ~0.89, best for capturing high-risk patients.

The final pipeline emphasizes recall, as missing a high-risk patient is more costly than a false alarm in the insurance context.

## GUI
- In you browser copy paste http://127.0.0.1:5000/predictdata. Input the parameters and press predict result button. 


## Clone repo
git clone https://github.com/your-username/diabetes-readmission-predictor.git
cd diabetes-readmission-predictor

### Install dependencies
pip install -r requirements.txt

### Run Flask app
python3 app.py



