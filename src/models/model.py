import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from pathlib import Path
import joblib

class Model:
    """Base abstract model"""
    def __init__(self, random_state, test_size, is_standard_split, is_save_model=False):
        self.random_state = random_state
        self.test_size = test_size
        self.is_standard_split = is_standard_split
        self.is_save_model = is_save_model

    def standard_split(self, X, y):
        """Split data into train and test sets."""
        return train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)

    def cross_validation(self, X, y):
        """Generator for cross-validation splits."""
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        for train_i, test_i in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train_i], y.iloc[train_i], X.iloc[test_i], y.iloc[test_i]
            yield (X_train, X_test, y_train, y_test)

    def train_model(self, X_train, y_train):
        raise NotImplementedError()

    def predict_model(self, X_test, model=None):
        raise NotImplementedError()

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Main training method for logistic regression."""
        model = self.train_model(X_train, y_train)
        y_pred = self.predict_model(X_test, model)
        return self.calculate_metrics(y_test, y_pred)

    def run_standard_split(self, X, y):
        """
        Perform standart split.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            list: Metrics for each fold
        """
        X_train, X_test, y_train, y_test = self.standard_split(X, y)
        return self.train_and_evaluate(X_train, X_test, y_train, y_test)

    def run_cross_validation(self, X, y):
        """
        Perform k-fold cross-validation.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            list: Metrics for each fold
        """
        metrics = []
        for X_train, X_test, y_train, y_test in self.cross_validation(
                X, y):
            metrics.append(self.train_and_evaluate(X_train, X_test, y_train, y_test))
        return metrics

    def accuracy(self, y_test, y_pred):
        """
        Compute accuracy score.

        Args:
            y_test: True labels
            y_pred: Predicted labels

        Returns:
            float: Accuracy score
        """
        y_test = np.array(y_test, dtype=float)
        tp_np = 0
        for i, j in zip(y_test, y_pred):
            if i == j:
                tp_np += 1
        return tp_np / len(y_pred)

    def recall(self, y_test, y_pred):
        """Compute recall score."""
        y_test = np.array(y_test, dtype=float)
        y_pred = np.array(y_pred, dtype=int)
        tp = 0
        fn = 0
        for i, j in zip(y_test, y_pred):
            if i == 1 and j == 1:
                tp += 1
            if (i == 1 and j == 0):
                fn += 1
        return tp / (tp + fn + 1e-10)
        '''
            Результаты моей метрики практически идеально совпадает с
            результатами оригинальной встроенной функции
            sklearn: 0.520694259012016
            custom:  0.520694259011946
        '''

    def precision(self, y_test, y_pred):
        """Compute precision score."""
        y_test = np.array(y_test, dtype=int)
        y_pred = np.array(y_pred, dtype=int)
        tp = 0
        fp = 0
        for i, j in zip(y_test, y_pred):
            if i == 1 and j == 1:
                tp += 1
            if i == 0 and j == 1:
                fp += 1
        return tp / (tp + fp + 1e-10)

    def f1(self, y_test, y_pred):
        """Compute f1 score."""
        y_test = np.array(y_test, dtype=float)
        y_pred = np.array(y_pred, dtype=int)
        pres = self.precision(y_test, y_pred)
        rec = self.recall(y_test, y_pred)
        f1 = (2 * pres * rec) / (pres + rec + 1e-10)
        return f1

    def print_weights(self, feature_names):
        if self.weights is None:
            print("Model haven't learned yet")
            return
        print("\nWeights:")
        print(f"Bias (w0): {self.weights[0]:.4f}")
        for name, w in zip(feature_names, self.weights[1:]):
            print(f"{name}: {w:.4f}")

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
            'accuracy': round(self.accuracy(y_test, y_pred), 4),
            'precision': round(self.precision(y_test, y_pred), 4),
            'recall': round(self.recall(y_test, y_pred), 4),
            'f1': round(self.f1(y_test, y_pred), 4),
            'ROC-AUC': round(roc_auc_score(y_test, y_pred), 4)
        }
        return metrics

    def average_metrics(self, metrics):
        """
        Calculate mean values of evaluation metrics across multiple runs/folds.

        Args:
            metrics_list (list of dict): List of metric dictionaries where each dict
                                        contains keys: 'accuracy', 'precision',
                                        'recall', 'f1', 'roc_auc'

        Returns:
            dict: Dictionary with averaged metrics in format:
                  {
                      'accuracy': float,
                      'precision': float,
                      'recall': float,
                      'f1': float,
                      'roc_auc': float
                  }
        """
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'ROC-AUC']
        return {
            metric: np.mean([m[metric] for m in metrics]) for metric in metric_names
        }

    def save_model(self, X, y, model_type):
        model = self.train_model(X, y)
        model_dir = Path(__file__).parent.parent.parent / 'api_files'
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f'{model_type}.joblib'
        joblib.dump(model, model_path)
