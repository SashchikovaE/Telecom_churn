from preprocessing import DataPreprocessor
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.logistic_regression_custom import LogisticRegressionCustom
from src.models.logistic_regression_original import LogisticRegressionOriginal
from src.models.random_forest_original import RandomForestOriginal

if __name__ == "__main__":
    non_coded_cols = [
        'customerID',
        'SeniorCitizen',
        'tenure',
        'MonthlyCharges',
        'TotalCharges']
    df_class = DataPreprocessor()
    df_class.preprocess(non_coded_cols)
    y = df_class.df['Churn']
    X = df_class.df.drop(columns=['customerID', 'Churn'])

    # Cross-validation sklearn RandomForest
    rf_orig_cv = RandomForestOriginal(n_estimators=100, max_depth=10, class_weight='balanced',
                                      random_state=42, test_size=0.3, is_standard_split=False, is_save_model=False)
    rf_orig_cv.run_randfor_orig(X, y)

    # Cross-validation custom LogisticRegression
    lr_cust_cv = LogisticRegressionCustom(penalty='l2', lambd=0.01, max_iter=100000, learning_rate=0.001,
                                          random_state=42, test_size=0.3, is_standard_split=False, is_save_model=False)
    lr_cust_cv.run_logreg_custom(X, y)

    # Cross-validation sklearn LogisticRegression
    lr_orig_cv = LogisticRegressionOriginal(penalty='l2', lambd=0.01, max_iter=100000, class_weight='balanced',
                                            random_state=42, test_size=0.3, is_standard_split=False, is_save_model=True)
    lr_orig_cv.run_logreg_orig(X, y)
