from src.data_preprocessing import load_data, preprocess_data, split_data
from src.model_training import train_model, evaluate_model, save_model
import joblib
import os

def main():
    os.makedirs("models", exist_ok=True)

    # Load and preprocess
    df = load_data("data/insurance.csv")
    df["bmi_smoker"] = df["bmi"] * (df["smoker"] == "yes").astype(int)

    X, y, preprocessor = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train and evaluate
    model = train_model(X_train, y_train)
    score = evaluate_model(model, X_test, y_test)
    print(f"Model R^2 score: {score:.4f}")

    # Save model and preprocessor
    save_model(model, "models/insurance_model.pkl")
    joblib.dump(preprocessor, "models/preprocessor.pkl")

if __name__ == "__main__":
    main()
