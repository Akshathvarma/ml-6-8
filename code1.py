import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

project_df = pd.read_excel("ecg_eeg_features.csv.xlsx")  
lab_df = pd.read_excel("Lab Session Data.xlsx")          
lab_df = lab_df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)']].dropna()

X_proj = project_df.drop(columns=['signal_type'])
y_proj = project_df['signal_type']
X_train, X_test, y_train, y_test = train_test_split(X_proj, y_proj, test_size=0.3, random_state=42)

model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_train, y_train)

y_pred_train = model_knn.predict(X_train)
y_pred_test = model_knn.predict(X_test)

print("A1: ECG Classification")
print("Train Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
print("Train Report:\n", classification_report(y_train, y_pred_train))
print("Test Report:\n", classification_report(y_test, y_pred_test))

X_lab = lab_df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]
y_lab = lab_df['Payment (Rs)']

reg_model = LinearRegression()
reg_model.fit(X_lab, y_lab)

y_pred_lab = reg_model.predict(X_lab)

print("\nA2: Price Prediction")
print("MSE:", mean_squared_error(y_lab, y_pred_lab))
print("RMSE:", np.sqrt(mean_squared_error(y_lab, y_pred_lab)))
print("MAPE:", mean_absolute_percentage_error(y_lab, y_pred_lab))
print("R² Score:", r2_score(y_lab, y_pred_lab))

np.random.seed(0)
train_X = np.random.uniform(1, 10, (20, 2))
train_y = np.array([0]*10 + [1]*10)

plt.figure()
plt.scatter(train_X[:10, 0], train_X[:10, 1], color='blue', label='Class 0')
plt.scatter(train_X[10:, 0], train_X[10:, 1], color='red', label='Class 1')
plt.legend()
plt.title("A3: Synthetic Training Data")
plt.grid(True)
plt.savefig("a3_training.png")
plt.close()

xx, yy = np.meshgrid(np.arange(0, 10.1, 0.1), np.arange(0, 10.1, 0.1))
grid = np.c_[xx.ravel(), yy.ravel()]

knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_3.fit(train_X, train_y)

Z = knn_3.predict(grid)

plt.figure()
plt.scatter(grid[:, 0], grid[:, 1], c=Z, cmap='coolwarm', alpha=0.3, s=10)
plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap='coolwarm', edgecolors='k')
plt.title("A4: Decision Boundary (k=3)")
plt.savefig("a4_k3.png")
plt.close()

for k in [1, 5, 10, 15]:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(train_X, train_y)
    Zk = knn_k.predict(grid)

    plt.figure()
    plt.scatter(grid[:, 0], grid[:, 1], c=Zk, cmap='coolwarm', alpha=0.3, s=10)
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap='coolwarm', edgecolors='k')
    plt.title(f"A5: Decision Boundary (k={k})")
    plt.savefig(f"a5_k{k}.png")
    plt.close()

proj_filtered = project_df[project_df['signal_type'].isin(['ECG', 'EEG'])].copy()
proj_filtered['label'] = proj_filtered['signal_type'].map({'ECG': 0, 'EEG': 1})

X_proj_2d = proj_filtered[['mean_val', 'entropy']].values
y_proj_2d = proj_filtered['label'].values

scaler = StandardScaler()
X_proj_2d_scaled = scaler.fit_transform(X_proj_2d)

knn_proj = KNeighborsClassifier(n_neighbors=3)
knn_proj.fit(X_proj_2d_scaled, y_proj_2d)

x_min, x_max = X_proj_2d_scaled[:, 0].min() - 1, X_proj_2d_scaled[:, 0].max() + 1
y_min, y_max = X_proj_2d_scaled[:, 1].min() - 1, X_proj_2d_scaled[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid_proj = np.c_[xx.ravel(), yy.ravel()]
Z_proj = knn_proj.predict(grid_proj).reshape(xx.shape)

proj_sampled = proj_filtered.sample(n=400, random_state=42)
X_sampled_scaled = scaler.transform(proj_sampled[['mean_val', 'entropy']])

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z_proj, cmap='coolwarm', alpha=0.3)
plt.scatter(X_sampled_scaled[:, 0], X_sampled_scaled[:, 1], c=proj_sampled['label'], cmap='coolwarm', edgecolors='k', s=40)
plt.title("A6: Project ECG vs EEG (Standardized)")
plt.xlabel("Standardized mean_val")
plt.ylabel("Standardized entropy")
plt.grid(True)
plt.tight_layout()
plt.savefig("a6_project_fixed.png")
plt.close()

param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)

print("\nA7: GridSearchCV Results")
print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)
print("Test Accuracy (best model):", test_score)