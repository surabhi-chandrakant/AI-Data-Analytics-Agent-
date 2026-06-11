import pandas as pd
import numpy as np

class EDAAgent:
    """Automated Exploratory Data Analysis"""
    
    def analyze(self, df):
        """Perform comprehensive EDA"""
        analysis = {
            'basic_stats': {},
            'missing_counts': {},
            'missing_percentages': {},
            'correlation_matrix': None,
            'skewness': {},
            'kurtosis': {},
            'unique_counts': {}
        }
        
        # Basic statistics
        analysis['basic_stats'] = df.describe(include='all').to_dict()
        
        # Missing values
        analysis['missing_counts'] = df.isnull().sum().to_dict()
        analysis['missing_percentages'] = (df.isnull().sum() / len(df) * 100).to_dict()
        
        # Unique values
        analysis['unique_counts'] = df.nunique().to_dict()
        
        # Skewness and Kurtosis for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            analysis['skewness'][col] = df[col].skew()
            analysis['kurtosis'][col] = df[col].kurtosis()
        
        # Correlation matrix
        if len(numeric_cols) > 1:
            analysis['correlation_matrix'] = df[numeric_cols].corr()
        
        # Data types
        analysis['dtypes'] = df.dtypes.astype(str).to_dict()
        
        # Memory usage
        analysis['memory_usage'] = df.memory_usage(deep=True).sum() / 1024**2  # in MB
        
        return analysis