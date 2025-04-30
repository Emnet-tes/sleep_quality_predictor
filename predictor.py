import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("data/sleep_data.csv")

# Clean column names just in case
df.columns = df.columns.str.strip().str.replace('\xa0', ' ', regex=True)

# Encode categorical features
label_encoders = {}
categorical_columns = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split blood pressure into systolic and diastolic
df[['systolic', 'diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)

# Define features and target
features = [
    'Age', 'Gender', 'Occupation', 'Sleep Duration',
    'Physical Activity Level', 'Stress Level',
    'BMI Category', 'Heart Rate', 'Daily Steps', 'Sleep Disorder',
    'systolic', 'diastolic'
]
X = df[features]
y = df['Quality of Sleep']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
print("Linear Regression MSE:", mean_squared_error(y_test, lr_preds))

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

pr = LinearRegression()
pr.fit(X_train_poly, y_train)
pr_preds = pr.predict(X_test_poly)
print("Polynomial Regression MSE:", mean_squared_error(y_test, pr_preds))

# Decision Tree Regression
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)
print("Decision Tree Regression MSE:", mean_squared_error(y_test, dt_preds))
