from preprocessing import DataPreprocessor
from logistic_regression_custom import LogisticRegressionCustom
from logistic_regression_original import LogisticRegressionOriginal
from random_forest_original import RandomForestOriginal

if __name__ == "__main__":
    exclude_cols = [
        'customerID',
        'SeniorCitizen',
        'tenure',
        'MonthlyCharges',
        'TotalCharges']
    df_class = DataPreprocessor()
    df_class.preprocess(exclude_cols)
    y = df_class.df['Churn']
    X = df_class.df.drop(columns=['customerID', 'Churn'])

    ## Standard split sklearn RandomForest
    #rf_orig = RandomForestOriginal(n_estimators=100, max_depth=10, random_state=42, test_size=0.4, is_standard_split=1)
    #rf_orig.run_randfor_orig(X, y)
#
    ## Cross-validation sklearn RandomForest
    #rf_orig_cv = RandomForestOriginal(n_estimators=100, max_depth=10, random_state=42, test_size=0.4, is_standard_split=0)
    #rf_orig_cv.run_randfor_orig(X, y)
#
    ## Cross-validation custom LogisticRegression
    #lr_cust_cv = LogisticRegressionCustom(
    #    penalty='l2',
    #    lambd=0.01,
    #    max_iter=100000,
    #    random_state=42,
    #    test_size=0.2,
    #    learning_rate=0.001,
    #    is_standard_split=0)
    #lr_cust_cv.run_logreg_custom(X, y)
#
    ## Standard split custom LogisticRegression
    #lr_cust = LogisticRegressionCustom(penalty='l2', lambd=0.01, max_iter=100000, random_state=42, test_size=0.2,
    #                                         learning_rate=0.001, is_standard_split=1)
    #lr_cust.run_logreg_custom(X, y)

    # Cross-validation sklearn LogisticRegression
    lr_orig_cv = LogisticRegressionOriginal(
        penalty='l2',
        lambd=0.01,
        max_iter=100000,
        class_weight='balanced',
        random_state=42,
        test_size=0.4,
        is_standard_split=0)
    lr_orig_cv.run_logreg_orig(X, y)

    # Standard split sklearn LogisticRegression
    lr_orig = LogisticRegressionOriginal(penalty='l2', lambd=0.01, max_iter=100000, class_weight='balanced',
                                            random_state=42, test_size=0.4, is_standard_split=1)
    lr_orig.run_logreg_orig(X, y)
