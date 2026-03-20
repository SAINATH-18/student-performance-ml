# student_score_predictor.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "student_data.csv")  # CSV in same folder

df = pd.read_csv(file_path)

# Drop student_id if exists
if 'student_id' in df.columns:
    df = df.drop(['student_id'], axis=1)

# Convert categorical columns to numbers
df = pd.get_dummies(df)

# Split features and target
X = df.drop("exam_score", axis=1)
y = df["exam_score"]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# -------------------------------
# Step 2: Interactive User Input
# -------------------------------
print("=== Student Exam Score Predictor ===")
print("Enter your details below:")

# Create a blank row with same columns
new_student = pd.DataFrame([[0]*X.shape[1]], columns=X.columns)

# Ask user for numeric values only
for col in X.columns:
    if new_student[col].dtype == 'int64' or new_student[col].dtype == 'float64':
        val = float(input(f"{col}: "))
        new_student[col] = val

# Predict exam score
predicted_score = model.predict(new_student)[0]
print(f"\nPredicted exam score: {predicted_score:.2f}")
