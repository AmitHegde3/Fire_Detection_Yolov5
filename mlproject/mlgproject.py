import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report
from sklearn.ensemble import StackingClassifier
import sqlite3

# Load the CSV file without headers
data = pd.read_csv('mlproject\predictions.csv', header=None)

# Display the first few rows to understand the structure
print("Initial data preview:")
print(data.head())

# Set the first row as header and reassign the dataframe
data.columns = data.iloc[0]
data = data[1:].reset_index(drop=True)

# Display the new header to confirm
print("\nColumns after setting the first row as header:")
print(data.columns)

# Rename columns
data.columns = ['x1', 'y1', 'x2', 'y2', 'area', 'aspect_ratio', 'size_label']

# Convert columns to appropriate data types
data = data.astype({
    'x1': float,
    'y1': float,
    'x2': float,
    'y2': float,
    'area': float,
    'aspect_ratio': float,
    'size_label': str
})

# Calculate the area
data['area'] = abs((data['x2'] - data['x1']) * (data['y2'] - data['y1']))

# Display the first few rows to verify the changes
print("\nData preview after renaming and type conversion:")
print(data.head())

# Calculate area median
median_area = data['area'].median()
print("\nMeadian Area: " , median_area)

# Label area column and train model
data['area_label'] = data['area'].apply(lambda x: 'SMALL' if x < median_area else 'BIG')
X = data[['x1', 'y1', 'x2', 'y2', 'aspect_ratio']]
y = data['area_label']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

# Hyperparameter Tuning for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5)
grid_search_rf.fit(X_train, y_train)
print("\nBest parameters for Random Forest:", grid_search_rf.best_params_)
best_rf = grid_search_rf.best_estimator_

# Hyperparameter Tuning for Gradient Boosting
param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
grid_search_gb = GridSearchCV(gb, param_grid_gb, cv=5)
grid_search_gb.fit(X_train, y_train)
print("\nBest parameters for Gradient Boosting:", grid_search_gb.best_params_)
best_gb = grid_search_gb.best_estimator_

# Cross-validation for Random Forest
cv_scores_rf = cross_val_score(best_rf, X_train, y_train, cv=5)
print("\nCross-validation scores for Random Forest:", cv_scores_rf)
print("\nMean CV accuracy for Random Forest:", cv_scores_rf.mean())

# Cross-validation for Gradient Boosting
cv_scores_gb = cross_val_score(best_gb, X_train, y_train, cv=5)
print("\nCross-validation scores for Gradient Boosting:", cv_scores_gb)
print("\nMean CV accuracy for Gradient Boosting:", cv_scores_gb.mean())

# Ensemble Methods with Stacking Classifier
estimators = [('rf', best_rf), ('gb', best_gb)]
stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier())
stacking_classifier.fit(X_train, y_train)
y_pred_stacking = stacking_classifier.predict(X_test)
print("\nStacking Classifier")
print(classification_report(y_test, y_pred_stacking, target_names=label_encoder.classes_))

# Storing in Database
conn = sqlite3.connect('predicted_data.db')
cursor = conn.cursor()

# Create a Table
create_table_query = """
CREATE TABLE IF NOT EXISTS predicted_data (
    id INTEGER PRIMARY KEY,
    x1 REAL,
    y1 REAL,
    x2 REAL,
    y2 REAL,
    aspect_ratio REAL,
    size_label TEXT,
    predicted_area_label TEXT
)
"""
cursor.execute(create_table_query)
conn.commit()

# Insert Predicted Data
predicted_data = list(zip(X_test['x1'], X_test['y1'], X_test['x2'], X_test['y2'], X_test['aspect_ratio'], label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(y_pred_stacking)))
insert_query = "INSERT INTO predicted_data (x1, y1, x2, y2, aspect_ratio, size_label, predicted_area_label) VALUES (?, ?, ?, ?, ?, ?, ?)"
cursor.executemany(insert_query, predicted_data)
conn.commit()

# Fetch Data from Database
fetch_query = "SELECT * FROM predicted_data LIMIT 10"
cursor.execute(fetch_query)
fetched_data = cursor.fetchall()

# Print Fetched Data in a Readable Format
print("\nFetched Data:")
print("{:<5} {:<10} {:<10} {:<10} {:<10} {:<15} {:<15} {:<20}".format("ID", "X1", "Y1", "X2", "Y2", "Aspect Ratio", "Size Label", "Predicted Area Label"))
for row in fetched_data:
    print("{:<5} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<15.2f} {:<15} {:<20}".format(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]))

# Close Connection
conn.close()
