import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('processed_data.csv')
df = df.sample(frac=1).reset_index(drop=True)
# Remove 'date' and 'symbol' columns
df = df.drop(['date', 'symbol'], axis=1)

# Assuming df is your DataFrame
columns_to_use = ['price', 'SMA_50', 'SMA_10d', 'gross_profit', 'other_expense', 'ratio']
available_columns = [col for col in columns_to_use if col in df.columns]

X = df[available_columns]
y = df['to_buy']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Create and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Get feature importances
importances = rf_model.feature_importances_
feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

# Print feature importances
print("Feature Importances:")
print(feature_importances)

# Plot feature importances
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar')
plt.title("Feature Importances in Random Forest Model")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()