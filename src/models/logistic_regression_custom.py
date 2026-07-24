import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.models.model import Model

class LogisticRegressionCustom (Model):
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

    def __init__(self, penalty, lambd, max_iter, learning_rate,random_state,
                 test_size, is_standard_split, is_save_model):
        """Initialize logistic regression model with specified parameters."""
        super().__init__(random_state, test_size, is_standard_split, is_save_model)
        self.penalty = penalty
        self.lambd = lambd
        self.max_iter = max_iter
        self.learning_rate = learning_rate

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
        weights = np.random.normal(scale=1, size=n)
        loss = []
        a = []
        z = 0
        prev_loss = float('inf')
        for i in range(self.max_iter):
            z = np.dot(X_train, weights)
            a = self.sigmoid(z)
            dw = (1 / l) * np.dot(X_train.T, a - y_train)
            if self.penalty == 'l1':
                dw += self.lambd * np.sign(weights)
            if self.penalty == 'l2':
                dw += 2 * self.lambd * weights
            weights -= self.learning_rate * dw
            a_clip = np.clip(a, 1e-10, 1 - 1e-10)
            cur_loss = np.mean(-(y_train * np.log(a_clip) +
                                 (1 - y_train) * np.log(1 - a_clip)))
            loss.append(cur_loss)
            if abs(prev_loss - cur_loss) < 1e-10:
                break
            prev_loss = cur_loss
        plt.plot(loss)
        plt.title('Loss function')
        file = Path(__file__).parent.parent.parent / 'images/loss.png'
        file.parent.mkdir(exist_ok=True)
        plt.savefig(file)
        plt.show()
        '''
            Функция потерь резко падает и стремится к нулю, значит на данном этапе обучение
            модели проходит корректно
        '''
        # self.print_weights(n)
        return weights

    def predict_model(self, X_test, weights):
        """
        Make predictions on new data.

        Args:
            X_test: Feature matrix for prediction

        Returns:
            ndarray: Predicted class labels (0 or 1)
        """
        X_test = np.array(X_test, dtype=float)
        X_test = np.insert(X_test, 0, 1, axis=1)
        z = np.dot(X_test, weights)
        model = self.sigmoid(z)
        res = (model >= 0.5).astype(int)
        return res

    def print_weights(self, weights, feature_names):
        if weights is None:
            print("Model haven't learned yet")
            return
        print("\nWeights:")
        print(f"Bias (w0): {weights[0]:.4f}")
        for name, w in zip(feature_names, weights[1:]):
            print(f"{name}: {w:.4f}")

    def run_logreg_custom(self, X, y):
        """Execute complete training and evaluation pipeline."""
        print("custom logistic regression")
        if self.is_standard_split == 1:
            print("standard split")
            print(self.run_standard_split(X, y), "\n")
        else:
            print("cross validation")
            metrics = self.run_cross_validation(X, y)
            print(self.average_metrics(metrics),"\n")
        if self.is_save_model:
            self.save_model(X, y, 'logistic_regression_custom')
