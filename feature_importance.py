from src.data_preprocessing import load_data, preprocess_data
import matplotlib.pyplot as plt
import shap
import joblib

model = joblib.load("models/insurance_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

df = load_data("data/insurance.csv")
df["bmi_smoker"] = df["bmi"] * (df["smoker"] == "yes").astype(int)
X, y, _ = preprocess_data(df)

explainer = shap.Explainer(model, X)
shap_values = explainer(X)

plt.tight_layout()
shap.summary_plot(shap_values, X, show=False)
plt.savefig("models/shap_summary.png")
