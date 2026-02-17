# =====================================================
# STEP 1: Import Required Libraries
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =====================================================
# STEP 2: Load the Dataset
# =====================================================

# Load student dataset from CSV file
data = pd.read_csv("student_data.csv")

# Display first 5 rows to verify data
print("First 5 rows of dataset:")
print(data.head())


# =====================================================
# STEP 3: Data Visualization (Exploratory Data Analysis)
# =====================================================

# Graph 1: Attendance vs Result
plt.scatter(data['Attendance'], data['Result'])
plt.xlabel("Attendance")
plt.ylabel("Result (1 = Pass, 0 = Fail)")
plt.title("Attendance vs Result")
plt.show()

# Graph 2: Mid Marks vs Result
plt.scatter(data['Mid'], data['Result'])
plt.xlabel("Mid Marks")
plt.ylabel("Result (1 = Pass, 0 = Fail)")
plt.title("Mid Marks vs Result")
plt.show()


# =====================================================
# STEP 4: Separate Features and Target Variable
# =====================================================

# Input features (Attendance, Quiz, Assignment, Mid)
X = data.drop("Result", axis=1)

# Output label (Pass/Fail)
y = data["Result"]

print("\nInput Features (X):")
print(X.head())

print("\nOutput Labels (y):")
print(y.head())


# =====================================================
# STEP 5: Split Dataset into Training and Testing Sets
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# =====================================================
# STEP 6: Create and Train KNN Model
# =====================================================

# Create KNN classifier with k = 3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model using training data
knn.fit(X_train, y_train)

print("\nKNN model trained successfully.")


# =====================================================
# STEP 7: Model Testing and Accuracy Calculation
# =====================================================

# Predict results for test data
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)


# =====================================================
# STEP 8: Model Evaluation
# =====================================================

# Display Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# =====================================================
# STEP 9: Prediction for a New Student
# =====================================================

# New student data format: [Attendance, Quiz, Assignment, Mid]
new_student = [[78, 7, 16, 34]]

prediction = knn.predict(new_student)

print("\nNew Student Data:", new_student)

if prediction[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")


# =====================================================
# STEP 10: Conclusion and Real-Life Application
# =====================================================

print("\n--- PROJECT CONCLUSION ---")
print("This project uses the K-Nearest Neighbors (KNN) algorithm to predict student results.")
print("Attendance, quizzes, assignments, and mid-term marks are used as input features.")
print("The model successfully predicts whether a student will PASS or FAIL.")

print("\n--- REAL-LIFE APPLICATION ---")
print("This system can be used by universities to identify weak students early.")
print("Teachers can provide extra support, counseling, or remedial classes before final exams.")
# End of knn_student_prediction.py