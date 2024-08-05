# Created by RayLi
from src.data.make_dataset import load_and_preprocess_data
from src.feature_engineering.build_features import create_dummy_vars
from src.models.train_models import train_logistic_regression
from src.models.train_models import random_forest
from src.models.predict_model import evaluate_model

if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "src/data/raw/credit.csv"
    df = load_and_preprocess_data(data_path)

    # Create dummy variables and separate features and target
    X, y = create_dummy_vars(df)

    # Train the logistic regression model
    model, X_test_scaled, y_test = train_logistic_regression(X, y)

    # Evaluate the model
    accuracy = evaluate_model(model, X_test_scaled, y_test)
    print(f"Logistic Regression Accuracy: {accuracy}")
    
    # Train the random forest model
    model, X_test_scaled, y_test = random_forest(X,y)
    
    # Evaluate the model
    accuracy = evaluate_model(model, X_test_scaled, y_test)
    print(f"Random Forest Accuracy: {accuracy}")
