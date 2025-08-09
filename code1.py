import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

df = pd.read_excel("ecg_eeg_features.csv.xlsx")  

df["sig_num"] = df["signal_type"].map({"ECG": 0, "EEG": 1})

X = df.drop(columns=["signal_type", "sig_num"])
y = df["sig_num"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def reg_one(Xtr, ytr, Xte, yte, col):
    mdl = LinearRegression().fit(Xtr[[col]], ytr)
    pred_tr = mdl.predict(Xtr[[col]])
    pred_te = mdl.predict(Xte[[col]])
    return mdl, pred_tr, pred_te

def metrics(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

def reg_all(Xtr, ytr, Xte, yte):
    mdl = LinearRegression().fit(Xtr, ytr)
    pred_tr = mdl.predict(Xtr)
    pred_te = mdl.predict(Xte)
    return mdl, pred_tr, pred_te

def kmeans_fit(Xtr, k):
    return KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xtr)

def cluster_metrics(X, labels):
    return {
        "Silhouette": silhouette_score(X, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, labels),
        "Davies-Bouldin": davies_bouldin_score(X, labels)
    }

def kmeans_scores(Xtr, ks):
    out = {}
    for k in ks:
        mdl = kmeans_fit(Xtr, k)
        out[k] = cluster_metrics(Xtr, mdl.labels_)
    return out

def elbow(Xtr, ks, sample_size=1000):
    if len(Xtr) > sample_size:
        Xs = Xtr.sample(sample_size, random_state=42)
    else:
        Xs = Xtr.copy()
    dist = []
    for k in ks:
        mdl = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
        dist.append(mdl.inertia_)
    plt.plot(ks, dist, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.show()

if __name__ == "__main__":
    feat = "mean_val"
    mdl1, pred_tr1, pred_te1 = reg_one(X_train, y_train, X_test, y_test, feat)
    print("\nOne Feature Regression Metrics:")
    print("Train:", metrics(y_train, pred_tr1))
    print("Test:", metrics(y_test, pred_te1))

    mdl_all, pred_tr_all, pred_te_all = reg_all(X_train, y_train, X_test, y_test)
    print("\nAll Features Regression Metrics:")
    print("Train:", metrics(y_train, pred_tr_all))
    print("Test:", metrics(y_test, pred_te_all))

    km2 = kmeans_fit(X_train, 2)
    print("\nClustering Metrics (k=2):", cluster_metrics(X_train, km2.labels_))

    k_vals = range(2, 6)
    print("\nKMeans Scores for Different k:")
    for k, sc in kmeans_scores(X_train, k_vals).items():
        print(f"k={k}:", sc)

    elbow(X_train, range(2, 10))
