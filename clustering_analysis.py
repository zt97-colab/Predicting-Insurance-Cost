from src.data_preprocessing import load_data, preprocess_data, perform_clustering
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import os

def run_clustering(filepath):
    os.makedirs("models", exist_ok=True)
    df = load_data(filepath)
    df["bmi_smoker"] = df["bmi"] * (df["smoker"] == "yes").astype(int)

    X_processed, _, _ = preprocess_data(df)
    cluster_labels, kmeans = perform_clustering(X_processed, n_clusters=3)

    df["cluster"] = cluster_labels
    print(df.groupby("cluster").mean(numeric_only=True))

    # Save visual
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="cluster", y="charges")
    plt.title("Insurance Charges by Cluster")
    plt.savefig("models/cluster_charges.png")
    plt.close()

    joblib.dump(kmeans, "models/kmeans_model.pkl")

if __name__ == "__main__":
    run_clustering("data/insurance.csv")
