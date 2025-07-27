import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("data/credit_data.csv")
df['Credit_Score'] = df['Credit_Score'].map({'Good': 1, 'Bad': 0})

X = df.drop("Credit_Score", axis=1)
y = df["Credit_Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Model trained with accuracy: {acc*100:.2f}%")

# Save model
with open("credit_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Model saved as credit_model.pkl")
