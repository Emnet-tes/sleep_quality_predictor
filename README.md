# Sleep Disorder Prediction

## Project Overview

This project uses machine learning algorithms to predict sleep disorders based on user health and lifestyle data. The dataset contains demographic, physiological, and behavioral features. The goal is to classify whether a person has a sleep disorder using models such as Random Forest, Support Vector Machine (SVM), and Logistic Regression.

---

## Features

* Gender
* Age
* Occupation
* Sleep Duration
* Physical Activity Level
* Stress Level
* BMI Category
* Heart Rate
* Daily Steps
* Systolic Blood Pressure
* Diastolic Blood Pressure
* Quality of Sleep (for other analyses)

---

## Dataset

The data is loaded from `sleep_data.csv`. The dataset contains some missing values in the `Sleep Disorder` column which are handled by filling with 'None'. Categorical variables are encoded using label encoding.

---

## Installation

Make sure you have Python 3.x installed. Then, install the required libraries:

```bash
pip install pandas scikit-learn matplotlib
```

---

## Usage

1. Load and preprocess the data (handle missing values and encode categorical features).
2. Split the data into training and testing sets.
3. Scale features using `StandardScaler`.
4. Train classification models: Random Forest, SVM, Logistic Regression.
5. Evaluate models using accuracy and classification reports.
6. Visualize performance with confusion matrices.
7. Predict sleep disorder for new user inputs after proper preprocessing and scaling.

---

## How to Predict Sleep Disorder for New Users

Prepare a DataFrame with the same feature columns (same order) used during training. Scale the user input with the trained scaler and use the trained Random Forest model (or any other model) for prediction.

---

## Example Prediction Code Snippet

```python
# Example user input
user_input_df = pd.DataFrame([{
    'Gender': encoded_gender,
    'Age': age,
    'Occupation': encoded_occupation,
    'Sleep Duration': sleep_duration,
    'Physical Activity Level': physical_activity_level,
    'Stress Level': stress_level,
    'BMI Category': encoded_bmi_category,
    'Heart Rate': heart_rate,
    'Daily Steps': daily_steps,
    'Systolic BP': systolic_bp,
    'Diastolic BP': diastolic_bp
}])

# Scale and predict
user_input_scaled = scaler.transform(user_input_df)
prediction = rf.predict(user_input_scaled)
print("Predicted Sleep Disorder:", prediction[0])
```

---

## Authors

* Emnet Teshome

---

## License

This project is licensed under the MIT License.

