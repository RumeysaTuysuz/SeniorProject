# Occupancy Detection - Centralized vs. Federated Learning Comparison

This project compares classical machine learning models and federated learning approaches for occupancy detection in a room.

## Dataset

The project uses a dataset with the following features:
- Temperature
- Humidity
- Light
- CO2
- HumidityRatio
- Occupancy (Target variable: 0 for not occupied, 1 for occupied)

The data is sourced from three files:
- `data/datatraining.txt` (Used for training all models)
- `data/datatest.txt` (Used for validation of all models)
- `data/datatest2.txt` (Used for final testing of all models)

## Features Implemented

1.  **Classical Machine Learning Models (Centralized Training):**
    *   Random Forest
    *   K-Nearest Neighbors (KNN)
    *   XGBoost
    *   Gaussian Naive Bayes
    *   Perceptron
    *   Support Vector Machine (SVM)
    *   Decision Tree (CART)

2.  **Federated Learning Algorithms (using Flower framework):**
    *   **FedAvg (Federated Averaging):** Standard federated learning algorithm.
    *   **FedProx:** A variation of FedAvg designed to handle system and data heterogeneity among clients by adding a proximal term to the local loss function.

    Federated learning simulations are run with varying numbers of clients (e.g., 2, 3, 5, 10) to observe the impact of client count on performance.

3.  **Evaluation Metrics:**
    *   Accuracy
    *   Confusion Matrix
    *   Training Time

4.  **Outputs and Visualizations:**
    *   **Comparative Results Table:** A summary table (logged to console and saved as `results/comparative_results_table.html`) comparing all models based on validation accuracy, test accuracy, and training time.
    *   **Confusion Matrices:** For the test set (`datatest2.txt`), confusion matrices are generated for each classical model and the final global federated learning models (saved in the `results` directory).
    *   **Learning Curves (Federated Learning):** For each federated learning strategy and client count, separate plots show:
        *   Validation accuracy and validation loss over federated rounds (client-side evaluation on `datatest.txt`).
        *   Test accuracy and test loss over federated rounds (server-side evaluation on `datatest2.txt`).
        (Saved in the `results` directory, e.g., `lc_validation_FedAvg_2clients.png`, `lc_test_FedAvg_2clients.png`).
    *   **Model Complexity/Convergence Curves (Classical Models):** For several classical models, plots show training and validation accuracy as a key hyperparameter (related to model complexity or convergence) is varied. This helps visualize overfitting.
        *   Decision Tree: `max_depth` vs. Accuracy
        *   KNN: `n_neighbors` vs. Accuracy
        *   Random Forest: `max_depth` vs. Accuracy
        *   XGBoost: `max_depth` vs. Accuracy
        *   SVM: `C` parameter vs. Accuracy
        *   Perceptron: `max_iter` vs. Accuracy
        (Saved in the `results` directory, e.g., `complexity_curve_decision_tree.png`).
