from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pickle
from pathlib import Path
import os

class RandomForestOriginal:

    def __init__(self, n_estimators, max_depth, random_state, test_size, is_standard_split):
        """
        Initialize logistic regression model with specified parameters.
        """
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
                Train model and evaluate on test set.

                Args:
                    X_train (DataFrame): Training features
                    y_train (Series): Training labels
                    X_test (DataFrame): Test features
                    y_test (Series): Test labels

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

    def save_model(self, X, y, model_type):
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.random_state, class_weight='balanced')
        model.fit(X, y)
        cur_file = Path(__file__)
        model_dir = cur_file.parent / 'models'
        model_path = model_dir / f'{model_type}.pkl'
        os.makedirs('models', exist_ok=True)
        with open(f'models/{model_type}.pkl', 'wb') as f:
            pickle.dump(model, f)

    def run_randfor_orig(self, X, y):
        print("sklearn random forest")
        if self.is_standard_split:
            print("standard split")
            print(self.run_standard_split(X, y), "\n")
        else:
            print("cross validation")
            metrics = self.run_cross_validation(X, y)
            print(self.average_metrics(metrics), "\n")
        self.save_model(X, y, 'random_forest_original')
