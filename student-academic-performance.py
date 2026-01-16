import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# LOAD DATA
# =========================
data = pd.read_csv('student_academic_performance_dataset.csv')

# Drop non-predictive column
if 'student_id' in data.columns:
    data = data.drop(columns=['student_id'])

# Encode gender
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})

# =========================
# HANDLE MISSING VALUES
# =========================
numeric_cols = [
    'age',
    'previous_score',
    'attendance_rate',
    'study_hours_per_week',
    'midterm_score',
    'quiz_average',
    'assignment_average',
    'library_visits_per_week',
    'online_course_time',
    'participation_score',
    'sleep_hours_per_day',
    'extracurricular_hours',
    'part_time_job_hours',
    'prerequisite_score',
    'attendance_impact',
    'study_efficiency',
    'total_assessment_score'
]

categorical_cols = [
    'parent_education_level',
    'family_income',
    'tutoring',
    'course_difficulty',
    'final_grade'
]

for col in numeric_cols:
    data[col] = data[col].fillna(data[col].mean())

for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

data['gender'] = data['gender'].fillna(data['gender'].mode()[0])

# =========================
# PREPARE TRAINING DATA
# =========================
X = data.drop(columns=['final_score'])
y = data['final_score']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN MODEL
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"\nMAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2 * 100:.2f}%")

# =========================
# USER INPUT
# =========================
def get_user_input():
    return {
        'gender': input("Gender (Male/Female): "),
        'age': float(input("Age: ")),
        'previous_score': float(input("Previous Score: ")),
        'attendance_rate': float(input("Attendance Rate (0–1 or %): ")),
        'study_hours_per_week': float(input("Study Hours per Week: ")),
        'midterm_score': float(input("Midterm Score: ")),
        'quiz_average': float(input("Quiz Average: ")),
        'assignment_average': float(input("Assignment Average: ")),
        'library_visits_per_week': float(input("Library Visits per Week: ")),
        'online_course_time': float(input("Online Course Time (hrs/week): ")),
        'participation_score': float(input("Participation Score: ")),
        'sleep_hours_per_day': float(input("Sleep Hours per Day: ")),
        'extracurricular_hours': float(input("Extracurricular Hours: ")),
        'part_time_job_hours': float(input("Part-time Job Hours: ")),
        'parent_education_level': input("Parent Education Level: "),
        'family_income': input("Family Income (Low/Medium/High): "),
        'tutoring': input("Tutoring (Yes/No): "),
        'course_difficulty': input("Course Difficulty (Easy/Medium/Hard): "),
        'prerequisite_score': float(input("Prerequisite Score: ")),
        'attendance_impact': float(input("Attendance Impact (0–1 or %): ")),
        'study_efficiency': float(input("Study Efficiency (0–1 or %): ")),
        'total_assessment_score': float(input("Total Assessment Score: ")),
        'final_grade': input("Final Grade (A/B/C/etc): ")
    }

user_data = get_user_input()
user_df = pd.DataFrame([user_data])

# =========================
# CLEAN USER INPUT
# =========================
text_cols = [
    'gender', 'parent_education_level',
    'family_income', 'tutoring',
    'course_difficulty', 'final_grade'
]

for col in text_cols:
    user_df[col] = user_df[col].astype(str).str.strip().str.title()

# Fix percentages
for col in ['attendance_rate', 'attendance_impact', 'study_efficiency']:
    if user_df[col].iloc[0] > 1:
        user_df[col] = user_df[col] / 100

# Encode gender
user_df['gender'] = user_df['gender'].map({'Male': 1, 'Female': 0})

# Fill missing safely
for col in numeric_cols:
    user_df[col] = user_df[col].fillna(data[col].mean())

for col in categorical_cols:
    user_df[col] = user_df[col].fillna(data[col].mode()[0])

# One-hot encode & align
user_df = pd.get_dummies(user_df, drop_first=True)
user_df = user_df.reindex(columns=X_train.columns, fill_value=0)

# =========================
# PREDICTION
# =========================
predicted_score = model.predict(user_df)[0]
print(f"\n🎯 Predicted Final Score: {predicted_score:.2f}")
