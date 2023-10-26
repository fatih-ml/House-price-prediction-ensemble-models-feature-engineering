import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

class FatihOutliers:
    '''
    There are various methods in this class to deal with outliers.
    instantiate the class with a dataframe such as:
    outlier_handler = FatihOutliers(df)
    '''
    
    def __init__(self, df) :
        self.df = df
        self.numeric_column_name_list = df.select_dtypes(include='number').columns
    
    def calculate_IQR(self, col, IQR_multi=3):
        '''
        It takes a column name and returns IQR as a float
        '''
        IQR_dict = {}
        IQR_dict['Q1'] = self.df[col].quantile(.25)
        IQR_dict['Q3'] = self.df[col].quantile(.75)
        IQR_dict['IQR'] = IQR_dict['Q3'] - IQR_dict['Q1']
        IQR_dict['IQR_lower_w'] = IQR_dict['Q1'] - IQR_multi*IQR_dict['IQR']
        IQR_dict['IQR_upper_w'] = IQR_dict['Q3'] + IQR_multi*IQR_dict['IQR']
        
        return IQR_dict
    
    def show_outliers_for_features(self, col, method='IQR', IQR_multi=3, treshold_high=0.99, treshold_low=0.01):
        '''
        return a dataframe from all outliers in selected column
        df is already given to object instance, and this method will take a column name
        methods are 'IQR' and 'tresholds' otherwise raise ValueError
        Defaults ---> method='IQR' | IQR_multi=3  |  treshold_high=0.99  |  treshold_low=0.01
        '''
        if method == 'IQR':
            IQR_dict = self.calculate_IQR(col, IQR_multi)
            outliers = self.df[(self.df[col] < IQR_dict['IQR_lower_w']) | (self.df[col] > IQR_dict['IQR_upper_w'])]
            return outliers
        
        elif method == 'tresholds':
            treshold_upper = self.df[col].quantile(treshold_high) 
            treshold_lower = self.df[col].quantile(treshold_low)
            outliers = self.df[(self.df[col] < treshold_lower) | (self.df[col] > treshold_upper)]
            return outliers
            
        else:
            raise ValueError(f"Invalid method. Supported methods: 'IQR' or 'tresholds'")
        if len(outliers)==0:
            return f'There are no outliers in {col} feature'
        
        
    def show_all_outliers(self, method='IQR', IQR_multi=3, treshold_high=0.99, treshold_low=0.01):
        '''
        return a dataframe with all outlier observations in given dataframe
        df is already given to object instance
        methods are 'IQR' and 'tresholds' otherwise raise ValueError
        Defaults ---> method='IQR' | IQR_multi=3  |  treshold_high=0.99  |  treshold_low=0.01
        '''

        outliers_dict = {}
        for col in self.numeric_column_name_list:
            outliers = self.show_outliers_for_features(col, method, IQR_multi, treshold_high, treshold_low)
            if not outliers.empty:
                outliers_dict[col] = outliers
        
        if outliers_dict:
            all_outliers_df = pd.concat(outliers_dict.values(), axis=0)
            all_outliers_df.drop_duplicates(inplace=True)
            return all_outliers_df
        else:
            return None
    
    def show_outlier_stats(self, method='IQR', IQR_multi=3, treshold_high=0.99, treshold_low=0.01):
        '''
        return a dataframe with existing outlier statistics
        df is already given to object instance
        methods are 'IQR' and 'tresholds' otherwise raise ValueError
        Defaults ---> method='IQR' | IQR_multi=3  |  treshold_high=0.99  |  treshold_low=0.01
        '''
        
        outlier_stats_dict = {}
        for col in self.numeric_column_name_list:
            outliers = self.show_outliers_for_features(col, method, IQR_multi, treshold_high, treshold_low)
            if len(outliers)>0:
                outlier_stats_dict[col] = len(outliers)
        
        outlier_stats_df = pd.DataFrame({'OutlierCount': outlier_stats_dict})
        total_row = pd.DataFrame({'OutlierCount': {'Total':np.sum(outlier_stats_df.OutlierCount)}})
        outlier_stats_df = pd.concat([outlier_stats_df, total_row])
        return outlier_stats_df

    
    def plot_boxplots_for_all_numeric(self): 
        numeric_df = self.df.select_dtypes(include='number')
        qty_numeric_features = len(self.numeric_column_name_list)
        ncols  = 6
        nrows = math.ceil(qty_numeric_features/ncols) 
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 15))
        axes = axes.flatten()  # this code transform axis from 2D to 1D 

        # Iterate through the columns and create box plots
        for i, col in enumerate(numeric_df.columns):
            numeric_df[col].plot.box(ax=axes[i])
            axes[i].set_title(col)

        # Hide any remaining empty subplots
        for i in range(len(numeric_df.columns), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()
        
    def plot_dtl_hist(self, col, IQR_multi=3, vline1=0.01, vline2=0.99):
        # Calculate statistics
        mean = self.df[col].mean()
        std = self.df[col].std()
        skewness = self.df[col].skew()
        percentiles = self.df[col].quantile([vline1, vline2])
        IQR_dict = self.calculate_IQR(col, IQR_multi)
        
        # Create histogram
        plt.figure(figsize=(15, 4))
        sns.histplot(self.df[col], kde=True)

        # Add vertical dashed lines at specified percentiles
        plt.axvline(x=percentiles[vline1], color='b', linestyle='--', label=f'{vline1*100}th percentile')
        plt.axvline(x=percentiles[vline2], color='b', linestyle='--', label=f'{vline2*100}th percentile')
        plt.axvline(x=IQR_dict['IQR_lower_w'], color='r', linestyle='--', label=f'{IQR_multi}IQR lower whisker')
        plt.axvline(x=IQR_dict['IQR_upper_w'], color='g', linestyle='--', label=f'{IQR_multi}IQR upper Whisker')        

        # Add text annotations
        plt.text(0.5, 0.9, f'Mean: {mean:.2f}', transform=plt.gca().transAxes)
        plt.text(0.5, 0.8, f'Standard Deviation: {std:.2f}', transform=plt.gca().transAxes)
        plt.text(0.5, 0.7, f'Skewness: {skewness:.2f}', transform=plt.gca().transAxes)

        # Set plot title and labels
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

        # Add legend
        plt.legend()

        # Show the plot
        plt.show()
        
#