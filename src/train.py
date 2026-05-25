import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


def get_models():

    models = {

        "Logistic Regression": LogisticRegression(),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            random_state=42
        ),

        "SVM": SVC(),

        "KNN": KNeighborsClassifier()
    }

    return models


def train_model(model, X_train, y_train):

    model.fit(X_train, y_train)

    return model


def cross_validate_models(models, X_train, y_train):

    results = {}

    for name, model in models.items():

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="accuracy"
        )

        mean_accuracy = np.mean(scores)

        results[name] = mean_accuracy

        print(f"{name}")

        print("Scores:", scores)

        print(f"Mean Accuracy: {mean_accuracy:.4f}")

        print("-" * 40)

    return results


def get_best_model(results):

    best_model = max(results, key=results.get)

    best_accuracy = results[best_model]

    print(f"\nBEST MODEL: {best_model}")

    print(f"BEST CV ACCURACY: {best_accuracy:.4f}")

    return best_model, best_accuracy


def perform_grid_search(X_train, y_train):

    svm = SVC(class_weight="balanced")

    param_grid = {

        "C": [0.1, 1, 10],

        "kernel": ["linear", "rbf"],

        "gamma": ["scale", "auto"]
    }

    grid_search = GridSearchCV(

        estimator=svm,

        param_grid=param_grid,

        cv=5,

        scoring="accuracy",

        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(
        f"Best Parameters: "
        f"{grid_search.best_params_}"
    )

    print(
        f"Best CV Accuracy: "
        f"{grid_search.best_score_:.4f}"
    )

    best_model = grid_search.best_estimator_

    return best_model