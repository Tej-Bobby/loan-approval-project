from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def evaluate_model(model, X_test, y_test):

    # Make Predictions
    y_pred = model.predict(X_test)

    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate Classification Report
    report = classification_report(y_test, y_pred)

    # Print Results
    print(f"Test Accuracy: {accuracy:.4f}")

    print("\nClassification Report:\n")

    print(report)

    return accuracy, report