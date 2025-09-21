from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from pathlib import Path
import joblib

class LogisticRegressionOriginal:
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
                 random_state, test_size, is_standard_split):
        """
        Initialize logistic regression model with specified parameters.
        """
        self.penalty = penalty
        self.lambd = lambd
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.test_size = test_size
        self.is_standard_split = is_standard_split

    def standard_split(self, X, y):
        return train_test_split(
            X, y, stratify=y, test_size=self.test_size, random_state=self.random_state)

    def cross_validation(self, X, y):
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        for train_i, test_i in kf.split(X):
            X_train, X_test, y_train, y_test = X.iloc[train_i], X.iloc[test_i], y.iloc[train_i], y.iloc[test_i]
            yield (X_train, X_test, y_train, y_test)

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
                Train model and evaluate on test set.

                Args:
                    X_train (DataFrame): Training features
                    y_train (Series): Training labels
                    X_test (DataFrame): Test features
                    y_test (Series): Test labels

                Returns:
                    dict: Evaluation metrics
                """
        model = LogisticRegression(penalty=self.penalty, C=1 / self.lambd, solver='saga', max_iter=self.max_iter, tol=1e-5,
                                   class_weight=self.class_weight, random_state=self.random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return self.calculate_metrics(y_test, y_pred)

    def run_standard_split(self, X, y):
        X_train, X_test, y_train, y_test = self.standard_split(X, y)
        return self.train_and_evaluate(X_train, X_test, y_train, y_test)

    def run_cross_validation(self, X, y):
        metrics = []
        for X_train, X_test, y_train, y_test in self.cross_validation(X, y):
            metrics.append(self.train_and_evaluate(X_train, X_test, y_train, y_test))
        return metrics

    def calculate_metrics(self, y_test, y_pred):
        metrics = {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred), 4),
            'recall': round(recall_score(y_test, y_pred), 4),
            'f1': round(f1_score(y_test, y_pred), 4),
            'ROC-AUC': round(roc_auc_score(y_test, y_pred), 4)
        }
        return metrics

    def average_metrics(self, metrics):
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'ROC-AUC']
        return {
            metric: np.mean([m[metric] for m in metrics]) for metric in metric_names
        }

    def save_model(self, X, y, model_type):
        model = LogisticRegression(penalty=self.penalty, C=1 / self.lambd, solver='saga', max_iter=self.max_iter, tol=1e-5,
                                   class_weight=self.class_weight, random_state=self.random_state)
        model.fit(X, y)
        cur_file = Path(__file__)
        model_dir = cur_file.parent.parent / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f'{model_type}.joblib'
        joblib.dump(model, model_path)

    def run_logreg_orig(self, X, y):
        print("sklearn version")
        if self.is_standard_split:
            print("standard split")
            print(self.run_standard_split(X, y), "\n")
        else:
            print("cross validation")
            metrics = self.run_cross_validation(X, y)
            print(self.average_metrics(metrics), "\n")
        self.save_model(X, y, 'log_regression_original')
