import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Dealing_missing_values:
    def __init__(self, df):
        """
        Initialize the Dealing_missing_values class with a DataFrame.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
        """
        self.df = df

    def mmm(self, columns, method='mean'):
        """
        Fill missing values in specified columns using a specified method.

        Parameters:
            columns (list): List of column names to apply the method.
            method (str): The method to fill missing values ('mean', 'median', or 'mode').
        """
        if method == 'mean':
            self.df[columns] = self.df[columns].fillna(self.df[columns].mean())
        elif method == 'median':
            self.df[columns] = self.df[columns].fillna(self.df[columns].median())
        elif method == 'mode':
            self.df[columns] = self.df[columns].fillna(self.df[columns].mode().iloc[0])
        else:
            raise ValueError("Invalid method. Use 'mean', 'median', or 'mode'.")

    def replace(self, replacements):
        """
        Replace missing values based on a provided dictionary.

        Parameters:
            replacements (dict): Dictionary of column names and replacement values.
        """
        self.df.fillna(replacements, inplace=True)

    def drop_rows(self, columns):
        """
        Drop rows containing missing values in specified columns.

        Parameters:
            columns (list): List of column names to check for missing values.
        """
        self.df.dropna(subset=columns, inplace=True)

    def drop_columns(self, columns):
        """
        Drop specified columns with missing values.

        Parameters:
            columns (list): List of column names to be dropped.
        """
        self.df.drop(columns=columns, inplace=True)

    def show_missing_values(self):
        """
        Show a bar graph of missing value counts sorted in ascending order.
        if there are no missing values, it prints accordingly.
        """
        if self.df.isnull().sum().any().any():
            missing_counts = self.df.isnull().sum().sort_values(ascending=False)
            missing_counts = missing_counts[missing_counts > 0]

            plt.figure(figsize=(8,5)) 
            ax = sns.barplot(x=missing_counts.index, y=missing_counts.values)
            plt.xticks(rotation=90)
            plt.xlabel('Features with Nan values')
            plt.ylabel('Nan Value Count')
            plt.title('Nan Value Counts by Features')
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
            plt.show()
        else:
            print('There are no missing values')
            
    def show_rows_with_missing_values(self):
        '''
        Filter rows with missing values.

        Returns:
            pd.DataFrame: DataFrame with rows containing missing values.
        '''
        return self.df[self.df.isnull().any(axis=1)]