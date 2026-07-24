from sklearn.ensemble import RandomForestClassifier
from src.models.model import Model

class RandomForestOriginal (Model):
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
    def __init__(self, n_estimators, max_depth, class_weight, random_state, test_size, is_standard_split, is_save_model):
        """Initialize logistic regression model with specified parameters."""
        super().__init__(random_state, test_size, is_standard_split, is_save_model=is_save_model)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
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
        model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                       random_state=self.random_state, class_weight=self.class_weight)
        model.fit(X_train, y_train)
        return model

    def predict_model(self, X_test, model):
        y_pred = model.predict(X_test)
        return y_pred

    def run_randfor_orig(self, X, y):
        print("sklearn random forest")
        if self.is_standard_split:
            print("standard split")
            print(self.run_standard_split(X, y), "\n")
        else:
            print("cross validation")
            metrics = self.run_cross_validation(X, y)
            print(self.average_metrics(metrics), "\n")
        if self.is_save_model:
            self.save_model(X, y, 'random_forest_original')
