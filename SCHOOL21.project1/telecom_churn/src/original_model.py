from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

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
    def __init__(self, penalty, lambd, max_iter, class_weight, random_state, test_size, is_standard_split):
        """
        Initialize logistic regression model with specified parameters.
        """
        self.cv_metrics = []
        self.penalty = penalty
        self.lambd = lambd
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.test_size = test_size
        self.is_standard_split = is_standard_split

    def calculate_metrics(self, y_test, y_pred):
        """
        Calculate all classification metrics.

        Args:
            y_test: True labels
            y_pred: Predicted labels

        Returns:
            dict: Dictionary of metrics (accuracy, precision, recall, f1, roc_auc)
        """
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred)
        }
        return metrics

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
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
        model = LogisticRegression(penalty=self.penalty, C=1/self.lambd, solver='saga', max_iter=self.max_iter, tol=1e-5,
                                   class_weight=self.class_weight, random_state=self.random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred)
        if self.is_standard_split == 0:
            self.cv_metrics.append(metrics)
        else:
            print(metrics)

        return metrics

    def average_metrics(self, metrics_list):
        """
        Compute average metrics across cross-validation folds.

        Args:
            metrics_list (list): List of metric dictionaries

        Returns:
            dict: Averaged metrics with same keys as input
        """
        return {
            'accuracy': np.mean([m['accuracy'] for m in metrics_list]),
            'precision': np.mean([m['precision'] for m in metrics_list]),
            'recall': np.mean([m['recall'] for m in metrics_list]),
            'f1': np.mean([m['f1'] for m in metrics_list]),
            'roc_auc': np.mean([m['roc_auc'] for m in metrics_list])
        }

    def run_original(self, X, y):
        """
        Execute complete training and evaluation pipeline.

        Args:
            X (DataFrame): Feature matrix
            y (Series): Target vector

        Outputs:
            Prints evaluation results to console
        """
        print("sklearn version")
        if self.is_standard_split == 0:
            print("cross-validation version")
            kf = KFold(n_splits=5)
            self.cv_metrics = []
            for fold, (train_i, test_i) in enumerate(kf.split(X), 1):
                X_train, y_train = X.iloc[train_i], y.iloc[train_i]
                X_test, y_test = X.iloc[test_i], y.iloc[test_i]
                fold_metrics = self.train_and_evaluate(X_train, y_train, X_test, y_test)
                print(f"fold {fold} ", fold_metrics)
            avg_metrics = self.average_metrics(self.cv_metrics)
            print(avg_metrics)
        else:
            print("standart split version")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, stratify=y, test_size=self.test_size, random_state=self.random_state)
            self.train_and_evaluate(X_train, y_train, X_test, y_test)
