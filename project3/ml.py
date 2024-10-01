import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import pickle

df = pd.read_csv('processed_data.csv')
df = df.sample(frac=1).reset_index(drop=True)
# Remove 'date' and 'symbol' columns
df = df.drop(['date', 'symbol'], axis=1)

# Separate features (X) and target variable (y)

columns_to_use = ['price', 'SMA_50', 'SMA_10d','gross_profit', 'other_expense', 'ratio']
available_columns = [col for col in columns_to_use if col in df.columns]


X = df[available_columns]
y = df['to_buy']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Neural Network': MLPClassifier(random_state=42, max_iter=1000),
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the XGBoost model
    if name == 'XGBoost':
        with open('xgboost_model.pkl', 'wb') as model_file:
            pickle.dump(model, model_file)
        print("XGBoost model saved as 'xgboost_model.pkl'")

# Save the scaler
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
print("Scaler saved as 'scaler.pkl'")

# Function to predict 'to_buy' for new data
def predict_to_buy(price, sma_50, gross_profit):
    return model.predict([[price, sma_50, gross_profit]])[0]

