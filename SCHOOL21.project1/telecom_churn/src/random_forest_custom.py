#from sklearn.model_selection import KFold, train_test_split
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
#import numpy as np#

#class RandomForestCustom:#

#    def __init__(self, random_state, test_size, is_standard_split):
#        """
#        Initialize logistic regression model with specified parameters.
#        """
#        self.random_state = random_state
#        self.test_size = test_size
#        self.is_standard_split = is_standard_split#

#    def standard_split(self, X, y):
#        return train_test_split(
#            X, y, stratify=y, test_size=self.test_size, random_state=self.random_state)#

#    def cross_validation(self, X, y):
#        kf = KFold(n_splits=5, shuffle=True, random_state=42)
#        for train_i, test_i in kf.split(X):
#            X_train, X_test, y_train, y_test = X.iloc[train_i], X.iloc[test_i], y.iloc[train_i], y.iloc[test_i]
#            yield (X_train, X_test, y_train, y_test)#

#    def train_model(self, X_train, y_train):#

#        return self.weights#

#    def train_and_evaluate(self, X_train, X_test, y_train, y_test):#

#        return self.calculate_metrics(y_test, y_pred)#

#    def run_standard_split(self, X, y):
#        X_train, X_test, y_train, y_test = self.standard_split(X, y)
#        return self.train_and_evaluate(X_train, X_test, y_train, y_test)#

#    def run_cross_validation(self, X, y):
#        metrics = []
#        for X_train, X_test, y_train, y_test in self.cross_validation(X, y):
#            metrics.append(self.train_and_evaluate(X_train, X_test, y_train, y_test))
#        return metrics#

#    def calculate_metrics(self, y_test, y_pred):
#        metrics = {
#            'accuracy': round(accuracy_score(y_test, y_pred), 4),
#            'precision': round(precision_score(y_test, y_pred), 4),
#            'recall': round(recall_score(y_test, y_pred), 4),
#            'f1': round(f1_score(y_test, y_pred), 4),
#            'ROC-AUC': round(roc_auc_score(y_test, y_pred), 4)
#        }
#        return metrics#

#    def average_metrics(self, metrics):
#        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'ROC-AUC']
#        return {
#            metric: np.mean([m[metric] for m in metrics]) for metric in metric_names
#        }#

#    def run_randfor_custom(self, X, y):
#        print("custom random forest")
#        if self.is_standard_split:
#            print("standard split")
#            print(self.run_standard_split(X, y), "\n")
#        else:
#            print("cross validation")
#            metrics = self.run_cross_validation(X, y)
#            print(self.average_metrics(metrics), "\n")
