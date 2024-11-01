# Importation des bibliothèques nécessaires
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import pandas as pd

# Charger les données du dataset Iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Construction de la pipeline
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),  # Standardisation
        (
            "classifier",
            RandomForestClassifier(random_state=42),
        ),  # Modèle de classification
    ]
)

# Définition des paramètres à explorer dans GridSearchCV
param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [None, 10, 20, 30],
    "classifier__min_samples_split": [
        2,
        5,
        10,
    ],  # Nombre minimum d'échantillons requis pour diviser un nœud
    "classifier__min_samples_leaf": [
        1,
        2,
        4,
    ],  # Nombre minimum d'échantillons dans une feuille
}

# Initialisation de GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)

# Démarrage du tracking MLflow
mlflow.set_experiment("RandomForest_Iris")

with mlflow.start_run():
    # Entraînement avec GridSearch et validation croisée
    grid_search.fit(X_train, y_train)

    # Prédiction sur les données de test
    y_pred = grid_search.best_estimator_.predict(X_test)

    # Mesure de la précision
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Affichage du rapport de classification
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Logging des hyperparamètres et des résultats dans MLflow
    mlflow.log_param(
        "best_n_estimators", grid_search.best_params_["classifier__n_estimators"]
    )
    mlflow.log_param(
        "best_max_depth", grid_search.best_params_["classifier__max_depth"]
    )
    mlflow.log_param(
        "best_min_samples_split",
        grid_search.best_params_["classifier__min_samples_split"],
    )
    mlflow.log_param(
        "best_min_samples_leaf",
        grid_search.best_params_["classifier__min_samples_leaf"],
    )

    mlflow.log_metric("accuracy", accuracy)

    # Enregistrer le modèle dans MLflow
    mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest_model")

    print("Modèle sauvegardé dans MLflow")
