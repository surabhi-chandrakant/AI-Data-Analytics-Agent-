import pandas as pd
import numpy as np
from scipy import stats

class DataCleaningAgent:
    """Automated data cleaning and preprocessing"""
    
    def clean(self, df):
        """Clean the dataset automatically"""
        report = {
            'initial_shape': df.shape,
            'initial_missing': int(df.isnull().sum().sum()),
            'initial_duplicates': int(df.duplicated().sum()),
            'actions': []
        }
        
        df_cleaned = df.copy()
        
        # Remove duplicates
        duplicates = df_cleaned.duplicated().sum()
        if duplicates > 0:
            df_cleaned = df_cleaned.drop_duplicates()
            report['actions'].append(f"Removed {duplicates} duplicate rows")
        
        # Handle missing values
        for col in df_cleaned.columns:
            missing = df_cleaned[col].isnull().sum()
            if missing > 0:
                if df_cleaned[col].dtype in ['int64', 'float64']:
                    # Numerical column - fill with median
                    median_val = df_cleaned[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    df_cleaned[col].fillna(median_val, inplace=True)
                    report['actions'].append(f"Filled {missing} missing values in '{col}' with median ({median_val:.2f})")
                else:
                    # Categorical column - fill with mode
                    mode_val = df_cleaned[col].mode()
                    if len(mode_val) > 0 and not pd.isna(mode_val[0]):
                        fill_value = mode_val[0]
                    else:
                        fill_value = "Unknown"
                    df_cleaned[col].fillna(fill_value, inplace=True)
                    report['actions'].append(f"Filled {missing} missing values in '{col}' with mode ({fill_value})")
        
        # Fix data types
        for col in df_cleaned.columns:
            # Try to convert numeric strings to numbers
            if df_cleaned[col].dtype == 'object':
                try:
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col])
                    report['actions'].append(f"Converted '{col}' to numeric type")
                except (ValueError, TypeError):
                    pass
            
            # Try to convert to datetime
            if col.lower() in ['date', 'datetime', 'timestamp', 'created_at', 'updated_at']:
                try:
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col])
                    report['actions'].append(f"Converted '{col}' to datetime type")
                except (ValueError, TypeError):
                    pass
        
        # Handle outliers using IQR method
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            try:
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)).sum()
                if outliers > 0:
                    # Cap outliers instead of removing
                    df_cleaned[col] = df_cleaned[col].clip(lower_bound, upper_bound)
                    report['actions'].append(f"Capped {outliers} outliers in '{col}'")
            except:
                pass
        
        report['final_shape'] = df_cleaned.shape
        report['final_missing'] = int(df_cleaned.isnull().sum().sum())
        report['final_duplicates'] = int(df_cleaned.duplicated().sum())
        
        return df_cleaned, report
    
    def assess_quality(self, df):
        """Assess data quality score"""
        score = 0
        total = 4
        
        # Completeness
        if df.isnull().sum().sum() == 0:
            score += 1
        
        # Uniqueness (no duplicates)
        if df.duplicated().sum() == 0:
            score += 1
        
        # Consistency (data types)
        try:
            if all(df[col].dtype in ['int64', 'float64', 'object', 'datetime64[ns]'] for col in df.columns):
                score += 1
        except:
            score += 0.5
        
        # Validity (no extreme outliers)
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            try:
                valid = True
                for col in numeric_cols:
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    if (z_scores > 3).sum() > len(df) * 0.05:  # More than 5% outliers
                        valid = False
                        break
                if valid:
                    score += 1
            except:
                score += 0.5
        
        return score / total if total > 0 else 0