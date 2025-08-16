from preprocessing import DataPreprocessor
from custom_model import LogisticRegressionCustom
from original_model import LogisticRegressionOriginal

if __name__ == "__main__":
    exclude_cols = [
        'customerID',
        'SeniorCitizen',
        'tenure',
        'MonthlyCharges',
        'TotalCharges']
    df_class = DataPreprocessor(
        r"C:\Users\Admin\PycharmProjects\pythonProject\SCHOOL21.project1\telecom_churn\data\telecom.csv")
    df_class.preprocess(exclude_cols)
    y = df_class.df['Churn']
    X = df_class.df.drop(columns=['customerID', 'Churn'])

    # Cross-validation custom version
    model_custom = LogisticRegressionCustom(
        penalty='l2',
        lambd=0.01,
        max_iter=100000,
        random_state=42,
        test_size=0.2,
        learning_rate=0.001,
        is_standard_split=0)
    model_custom.run_custom(X, y)

    # Standard split custom version
    model_custom1 = LogisticRegressionCustom(penalty='l2', lambd=0.01, max_iter=100000, random_state=42, test_size=0.2,
                                             learning_rate=0.001, is_standard_split=1)
    model_custom1.run_custom(X, y)

    # Cross-validation sklearn version
    model_orig = LogisticRegressionOriginal(
        penalty='l2',
        lambd=0.01,
        max_iter=100000,
        class_weight='balanced',
        random_state=42,
        test_size=0.4,
        is_standard_split=0)
    model_orig.run_original(X, y)

    # Standard split sklearn version
    model_orig = LogisticRegressionOriginal(penalty='l2', lambd=0.01, max_iter=100000, class_weight='balanced',
                                            random_state=42, test_size=0.4, is_standard_split=1)
    model_orig.run_original(X, y)
