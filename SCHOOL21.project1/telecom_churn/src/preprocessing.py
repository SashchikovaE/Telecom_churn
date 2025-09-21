import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

class DataPreprocessor:
    """
    A class for preprocessing telecom customer churn data.

    This class handles all data preparation steps including:
    - Feature engineering
    - Categorical encoding
    - Missing value treatment
    - Outlier detection
    - Data normalization
    - Visualization
    Attributes:
        df (pd.DataFrame): The dataframe containing the loaded and processed data.
    """

    def __init__(self, df=None):
        """
        Initialize the DataPreprocessor with data from a CSV file.

        Args:
            filepath (str): Path to the CSV file containing the raw data.
        """
        if df is None:
            data_path = Path(__file__).parent.parent / 'data' / 'telecom.csv'
            self.df = pd.read_csv(str(data_path))
        else:
            self.df = df

    def encode_categorical_data(self, col, mapping, is_binary_category=0, drop_first=True):
        """
        Encode categorical features into numerical values

        Args:
            col (str): Column name to encode.
            mapping (dict): Dictionary for value mapping.
            is_binary_category (bool): If True, applies simple mapping.
                                      If False, performs one-hot encoding.
        """
        if is_binary_category == 0:
            dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=drop_first)
            dummies = dummies.astype(int)
            self.df = pd.concat([
                self.df.drop(col, axis=1),
                dummies,
            ], axis=1)
        if is_binary_category == 1:
            self.df[col] = self.df[col].map(mapping)

    def convert_object_to_float(self):
        """
        Convert object-type columns to float where possible

        Handles columns that contain numeric values stored as strings.
        """
        for col in self.df.select_dtypes(include=['object']).columns:
            if col != self.df.columns[0]:
                self.df[col] = pd.to_numeric(
                    self.df[col].astype(str), errors='coerce')

    def analyze_empty_rows_and_cols(self):
        """
        Analyze and handle missing values in the dataset.

        Replaces NaN values with 0 based on business logic.
        """
        pd.set_option('display.width', None)
        print(self.df[self.df.isna().any(axis=1)])
        '''
            Так как tenure = 0 и Churn_encoded = 0, то есть не произошло никаких
            ошибок, клиент меньше месяца пользуется данными услугами и ни один клиент не ушел,
            можно сделать вывод, что эти клиенты только оформили подписку, и еще не пришло время оплаты:
            таким образом, можем заменить все Nan в столбце TotalCharges нулями.
        '''
        self.df = self.df.fillna(0)
        print(self.df.isnull().sum())

    def check_outliers(self, cols):
        """
        Detect outliers using the IQR method.

        Args:
            cols (list): List of numeric columns to check for outliers.

        Prints:
            Number of outliers and outlier rows.
        """
        numeric_cols = self.df[cols].columns.tolist()
        stat = self.df[numeric_cols].describe()
        q1 = stat.loc['25%']
        q3 = stat.loc['75%']
        iqr = q3 - q1
        lower_bound = q1 - 0.5 * iqr
        upper_bound = q3 + 0.5 * iqr
        outliers_mask = (
            self.df[numeric_cols] < lower_bound) | (
            self.df[numeric_cols] > upper_bound)
        anomalous_rows = self.df[outliers_mask.any(axis=1)]
        print(len(anomalous_rows))
        print(anomalous_rows)
        '''
            Из вывода столбцов с выбросами можно сделать вывод, что выбросов у численных столбцов нет,
            и мы можем работать дальше спокойно
        '''

    def visualize_outliers(self):
        """
        Visualize outliers using boxplots for key numeric columns.

        Saves plot to 'images/outliers_boxplot.png'.
        """
        sns.boxplot(
            data=self.df[['tenure', 'MonthlyCharges', 'TotalCharges']], whis=0.5)
        '''
            Уменьшив коэффициент IQR до 0.5, получаем выбросы, а со стандартным коэффициентом -
            данные без выбросов. Проанализировав данные вручную, делаем вывод, что выбросы в столбце
            'TotalCharges' обусловлены высокой стоимостью тарифа и длительным использованием услуг.
        '''
        plt.title('Analyze of outliers')
        plt.savefig('images/outliers_boxplot.png')
        plt.show()
        plt.close()

    def add_new_features(self, is_tenure_status=1):
        """
        Add new engineered features to the dataset.

        Args:
            is_tenure_status (bool): If True, adds tenure_status feature (years of service).
                                     If False, adds avg_monthly_charges feature.
        """
        if is_tenure_status == 1:
            self.df['tenure_status'] = self.df['tenure'] // 12
        else:
            self.df['avg_monthly_charges'] = self.df['TotalCharges'] / \
                (self.df['tenure'] + 1e-10)


    def visualize_churn_distribution(self):
        """
        Visualize the distribution of churn classes.

        Saves plot to 'images/churn_distribution.png'.
        """
        sns.countplot(data=self.df, x='Churn')
        plt.title('Churn distribution of client')
        plt.savefig('images/churn_distribution.png')
        plt.show()
        '''
            Исходя из графика, выяснилось, что отношение оставшихся клиентов к ушедшим - 2.5 / 1 -
            что является допустимой разницей
        '''

    def visualize_correlation_matrix(self):
        """
        Visualize correlation matrix for numeric features.

        Saves plot to 'images/correlation_matrix.png'.
        """
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64'])
        sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f")
        plt.title('Correlation matrix')
        plt.savefig('images/correlation_matrix.png')
        plt.show()

    def normalize_data(self):
        """
        Normalize numeric features using Z-score normalization.

        Returns:
            pd.DataFrame: The dataframe with normalized numeric features.
        """
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        non_binary_cols = [
            col for col in numeric_cols if self.df[col].nunique() > 2]
        mean = self.df[non_binary_cols].mean(axis=0)
        std = self.df[non_binary_cols].std(axis=0)
        self.df[non_binary_cols] = (
            self.df[non_binary_cols] - mean) / (std + 1e-10)
        return self.df

    def print_table(self):
        """Print the current state of the dataframe."""
        pd.set_option('display.width', None)
        print(self.df)

    def preprocess(self, exclude_cols):
        """
        Execute the complete preprocessing pipeline.

        Args:
            exclude_cols (list): Columns to exclude from preprocessing.

        Returns:
            pd.DataFrame: The fully processed dataframe.
        """
        for i in self.df.columns.tolist():
            if i not in exclude_cols:
                print(i)
                print(self.df[i].unique())
                if len(self.df[i].unique()) > 2:
                    self.encode_categorical_data(
                        i, mapping={'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
                else:
                    self.encode_categorical_data(
                        i,
                        mapping={
                            'Yes': 1,
                            'No': 0,
                            'Male': 1,
                            'Female': 0},
                        is_binary_category=1)
        self.convert_object_to_float()
        self.analyze_empty_rows_and_cols()
        print(self.df.describe())
        self.check_outliers(cols=['tenure', 'MonthlyCharges', 'TotalCharges'])
        self.visualize_outliers()
        self.add_new_features()
        self.add_new_features(is_tenure_status=0)
        self.visualize_churn_distribution()
        self.visualize_correlation_matrix()
        self.print_table()
        self.normalize_data()
        self.print_table()
