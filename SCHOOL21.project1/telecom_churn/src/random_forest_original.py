from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

class RandomForestOriginal:
    """
       A wrapper class for sklearn's Random Forest classifier with enhanced evaluation capabilities.

       This class provides a simplified interface for training and evaluating Random Forest models
       with both standard train-test split and cross-validation approaches. It automatically
       calculates multiple evaluation metrics and supports class balancing.

       Attributes:
           n_estimators (int): The number of trees in the forest.
           max_depth (int): The maximum depth of the trees.
           random_state (int): Random state for reproducibility.
           test_size (float): Proportion of dataset to include in test split (0.0 to 1.0).
           is_standard_split (bool): If True, use standard train-test split;
                                    if False, use cross-validation.
    """
    def __init__(self, n_estimators, max_depth, random_state, test_size, is_standard_split):
        """Initialize logistic regression model with specified parameters."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
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
        Train model and evaluate on test set

        Args:
            X_train (DataFrame): Training features
            y_train (Series): Training labels
            X_test (DataFrame): Test features
            y_test (Series): Test label

        Returns:
            dict: Evaluation metrics
        """
        model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state, class_weight='balanced')
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

    def run_randfor_orig(self, X, y):
        print("sklearn random forest")
        if self.is_standard_split:
            print("standard split")
            print(self.run_standard_split(X, y), "\n")
        else:
            print("cross validation")
            metrics = self.run_cross_validation(X, y)
            print(self.average_metrics(metrics), "\n")
