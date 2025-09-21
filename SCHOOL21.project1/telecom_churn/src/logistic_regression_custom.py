import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import pickle
from pathlib import Path
import os

class LogisticRegressionCustom:
    """
    Custom implementation of logistic regression from scratch.

    This class provides:
    - Logistic regression with L1/L2 regularization
    - Custom implementation of evaluation metrics
    - Support for both standard train-test split and cross-validation
    - Training visualization

    Args:
        penalty (str): Regularization type ('l1' or 'l2')
        lambd (float): Regularization strength
        max_iter (int): Maximum number of iterations
        random_state (int): Random seed for reproducibility
        test_size (float): Proportion of test set (0.0-1.0)
        learning_rate (float): Learning rate for gradient descent
        is_standard_split (bool): If True uses train-test split, else uses CV
    """

    def __init__(self, penalty, lambd, max_iter, random_state,
                 test_size, learning_rate, is_standard_split):
        """
        Initialize logistic regression model with specified parameters.
        """
        self.weights = None
        self.penalty = penalty
        self.lambd = lambd
        self.max_iter = max_iter
        self.random_state = random_state
        self.test_size = test_size
        self.learning_rate = learning_rate
        self.is_standard_split = is_standard_split

    def standard_split(self, X, y):
        """
        Split data into train and test sets.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)

    def cross_validation(self, X, y):
        """
        Generator for cross-validation splits.

        Args:
            X: Feature matrix
            y: Target vector

        Yields:
            tuple: (X_train, X_test, y_train, y_test) for each fold
        """
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        for train_i, test_i in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train_i], y.iloc[train_i], X.iloc[test_i], y.iloc[test_i]
            yield (X_train, X_test, y_train, y_test)

    def sigmoid(self, z):
        """
        Compute sigmoid function with numerical stability.

        Args:
            z: Input value(s)

        Returns:
            float or ndarray: Sigmoid output in range [0, 1]
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def train_model(self, X_train, y_train):
        """
        Train model and evaluate on test set.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets

        Returns:
            tuple: (weights, X_test, y_test)
        """
        X_train = np.array(X_train, dtype=float)
        y_train = np.array(y_train, dtype=float)
        X_train = np.insert(X_train, 0, 1, axis=1)
        l, n = X_train.shape
        np.random.seed(self.random_state)
        self.weights = np.random.normal(scale=1, size=n)
        loss = []
        a = []
        z = 0
        prev_loss = float('inf')
        for i in range(self.max_iter):
            z = np.dot(X_train, self.weights)
            a = self.sigmoid(z)
            dw = (1 / l) * np.dot(X_train.T, a - y_train)
            if self.penalty == 'l1':
                dw += self.lambd * np.sign(self.weights)
            if self.penalty == 'l2':
                dw += 2 * self.lambd * self.weights
            self.weights -= self.learning_rate * dw
            a_clip = np.clip(a, 1e-10, 1 - 1e-10)
            cur_loss = np.mean(-(y_train * np.log(a_clip) +
                                 (1 - y_train) * np.log(1 - a_clip)))
            loss.append(cur_loss)
            if abs(prev_loss - cur_loss) < 1e-10:
                break
            prev_loss = cur_loss
        plt.plot(loss)
        plt.title('Loss function')
        plt.savefig('images/loss.png')
        plt.show()
        '''
            Функция потерь резко падает и стремится к нулю, значит на данном этапе обучение
            модели проходит корректно
        '''
        # self.print_weights(n)
        return self.weights

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Main training method for logistic regression.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            tuple: (weights, X_test, y_test) or (weights, None, None) for CV
        """
        self.weights = self.train_model(X_train, y_train)
        y_pred = self.predict(X_test)
        return self.calculate_metrics(y_test, y_pred)

    def predict(self, X_test):
        """
        Make predictions on new data.

        Args:
            X_test: Feature matrix for prediction

        Returns:
            ndarray: Predicted class labels (0 or 1)
        """
        X_test = np.array(X_test, dtype=float)
        X_test = np.insert(X_test, 0, 1, axis=1)
        z = np.dot(X_test, self.weights)
        model = self.sigmoid(z)
        res = (model >= 0.5).astype(int)
        return res

    def run_standard_split(self, X, y):
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
        """
        Compute recall score.

        Args:
            y_test: True labels
            y_pred: Predicted labels

        Returns:
            float: Recall score
        """
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
            custom:  0.5206942590119464
        '''

    def precision(self, y_test, y_pred):
        """
        Compute precision score.

        Args:
            y_test: True labels
            y_pred: Predicted labels

        Returns:
            float: Precision score
        """
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
        """
        Compute f1 score.

        Args:
            y_test: True labels
            y_pred: Predicted labels

        Returns:
            float: Precision score
        """
        y_test = np.array(y_test, dtype=float)
        y_pred = np.array(y_pred, dtype=int)
        pres = self.precision(y_test, y_pred)
        rec = self.recall(y_test, y_pred)
        f1 = (2 * pres * rec) / (pres + rec + 1e-10)
        return f1
        # y_test = np.array(y_test).astype(int)
        # y_pred = np.array(y_pred).astype(int)
        # return f1_score(y_test, y_pred)

    def print_weights(self, feature_names):
        """
        Print model weights with corresponding feature names in readable format.

        Displays the bias term (intercept) followed by each feature's weight.
        Prints a message if the model hasn't been trained yet.

        Args:
            feature_names (list): List of feature names corresponding to weights[1:]

        Output:
            Prints formatted weights to stdout, e.g.:
            Bias (w0): 0.1234
            feature1: -0.5678
            feature2: 0.9012
        """
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
        cur_file = Path(__file__)
        model_dir = cur_file.parent.parent.parent / 'models'
        model_path = model_dir / f'{model_type}.pkl'
        os.makedirs('models', exist_ok=True)
        with open(f'models/{model_type}.pkl', 'wb') as f:
            pickle.dump(model, f)

    def run_logreg_custom(self, X, y):
        """
        Execute complete training and evaluation pipeline.

        Args:
            X: Feature matrix
            y: Target vector

        Prints:
            Evaluation metrics for the model.
        """
        print("custom version")
        if self.is_standard_split == 1:
            print("standard split")
            print(self.run_standard_split(X, y), "\n")
        else:
            print("cross validation")
            metrics = self.run_cross_validation(X, y)
            print(self.average_metrics(metrics),"\n")
        self.save_model(X, y, 'log_regression_custom')
