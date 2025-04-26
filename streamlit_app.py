import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import gaussian_kde

# Load base model and preprocessor
base_model = joblib.load("models/insurance_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

st.set_page_config(page_title="Insurance Cost Estimator", layout="centered")
st.title("🧠 Insurance Predictor")
st.markdown("""
Enter patient details on the sidebar to predict insurance cost using intelligent analysis, trend forecasting, and dynamic learning features.
""")

# Sidebar input
st.sidebar.header("🔧 Input Patient Details")
age = st.sidebar.slider("Age", 18, 64, 30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.slider("BMI", 15.0, 45.0, 25.0)
children = st.sidebar.selectbox("Children", [0, 1, 2, 3, 4, 5])
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

input_df = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region],
})

input_df["bmi_smoker"] = input_df["bmi"] * (input_df["smoker"] == "yes").astype(int)

if st.sidebar.button("📊 Predict"):
    X_processed = preprocessor.transform(input_df)
    prediction = base_model.predict(X_processed)[0]

    st.subheader("💰 Predicted Insurance Cost")
    st.success(f"${prediction:,.2f}")

    try:
        kmeans = joblib.load("models/kmeans_model.pkl")
        cluster = kmeans.predict(X_processed)[0]
        st.subheader("🧠 Risk Cluster")
        if cluster == 2:
            st.error("⚠️ High Risk")
        elif cluster == 1:
            st.warning("🟠 Medium Risk")
        else:
            st.success("✅ Low Risk")
    except Exception as e:
        st.info(f"Run clustering_analysis.py to enable risk group prediction. Error: {e}")

    try:
        st.subheader("📈 Model Comparison")
        data = pd.read_csv("data/insurance.csv")
        data["bmi_smoker"] = data["bmi"] * (data["smoker"] == "yes").astype(int)
        X = data.drop("charges", axis=1)
        y = data["charges"]
        X_transformed = preprocessor.transform(X)

        lin_model = LinearRegression().fit(X_transformed, y)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_transformed, y)

        lin_pred = lin_model.predict(X_transformed)
        rf_pred = rf_model.predict(X_transformed)

        scores = pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest"],
            "R^2 Score": [r2_score(y, lin_pred), r2_score(y, rf_pred)],
            "MAE": [mean_absolute_error(y, lin_pred), mean_absolute_error(y, rf_pred)],
            "RMSE": [mean_squared_error(y, lin_pred) ** 0.5, mean_squared_error(y, rf_pred) ** 0.5]
        })

        st.dataframe(scores.round(2))
    except Exception as e:
        st.warning(f"Model comparison unavailable: {e}")

    # Einstein-Grade Curved Visuals
    try:
        st.subheader("🔬 Advanced Visual Trends Across All Features")
        df = pd.read_csv("data/insurance.csv")
        df["bmi_smoker"] = df["bmi"] * (df["smoker"] == "yes").astype(int)
        X_all = df.drop("charges", axis=1)
        X_transformed_all = preprocessor.transform(X_all)
        df["predicted_cost"] = base_model.predict(X_transformed_all)

        features = ["age", "bmi", "children", "bmi_smoker"]

        for feature in features:
            fig, ax = plt.subplots(figsize=(8, 5))
            x_vals = df[feature]
            y_vals = df["predicted_cost"]

            # Kernel Density Estimation for smoothing
            xy = np.vstack([x_vals, y_vals])
            z = gaussian_kde(xy)(xy)
            idx = np.argsort(x_vals)
            x_sorted, y_sorted, z_sorted = x_vals[idx], y_vals[idx], z[idx]

            sc = ax.scatter(x_sorted, y_sorted, c=z_sorted, cmap="viridis", s=20)
            ax.set_title(f"Predicted Insurance Cost vs {feature.capitalize()}")
            ax.set_xlabel(feature.capitalize())
            ax.set_ylabel("Predicted Cost ($)")
            plt.colorbar(sc, label="Data Density")
            st.pyplot(fig)

        # Smoker comparison
        st.subheader("🚬 Smoker vs Non-Smoker Cost Insights")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(data=df, x="smoker", y="predicted_cost", palette="coolwarm")
        ax.set_title("Smoker vs Non-Smoker: Predicted Insurance Costs")
        st.pyplot(fig)

        # Region-wise analysis
        st.subheader("🌎 Regional Cost Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(data=df, x="region", y="predicted_cost", palette="Set2")
        ax.set_title("Predicted Cost by Region")
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"Trend plots unavailable: {e}")