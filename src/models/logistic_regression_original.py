from sklearn.linear_model import LogisticRegression
from pathlib import Path
import joblib
from src.models.model import Model

class LogisticRegressionOriginal (Model):
    """
    Wrapper for sklearn's Logistic Regression with extended functionality.

    Provides standardized interface for:
    - Both standard train-test split and cross-validation
    - Consistent metrics calculation
    - Results comparison with custom implementation

    Args:
        penalty (str): 'l1' or 'l2' regularization
        lambd (float): Inverse regularization strength (smaller = stronger)
        max_iter (int): Maximum iterations for solver
        class_weight (str/dict): Weights for classes ('balanced' or dict)
        random_state (int): Random seed for reproducibility
        test_size (float): Test set proportion (0.0-1.0)
        is_standard_split (bool): True for train-test split, False for CV
    """

    def __init__(self, penalty, lambd, max_iter, class_weight,
                 random_state, test_size, is_standard_split, is_save_model):
        """Initialize logistic regression model with specified parameters."""
        super().__init__(random_state, test_size, is_standard_split, is_save_model)
        self.penalty = penalty
        self.lambd = lambd
        self.max_iter = max_iter
        self.class_weight = class_weight

    def train_model(self, X_train, y_train):
        """
        Train model and evaluate on test set

        Args:
            X_train (DataFrame): Training features
            y_train (Series): Training labels
            X_test (DataFrame): Test features
            y_test (Series): Test label

        Returns:
            dict: Evaluation metrics
        """
        model = LogisticRegression(penalty=self.penalty, C=1 / self.lambd, solver='saga', max_iter=self.max_iter, tol=1e-5,
                                   class_weight=self.class_weight, random_state=self.random_state)
        model.fit(X_train, y_train)
        return model

    def predict_model(self, X_test, model):
        y_pred = model.predict(X_test)
        return y_pred

    def run_logreg_orig(self, X, y):
        print("sklearn logistic regression")
        if self.is_standard_split:
            print("standard split")
            print(self.run_standard_split(X, y), "\n")
        else:
            print("cross validation")
            metrics = self.run_cross_validation(X, y)
            print(self.average_metrics(metrics), "\n")
        if self.is_save_model:
            self.save_model(X, y, 'log_regression_original')
