import pandas as pd
import numpy as np
from scipy import stats

class InsightAgent:
    """Generate actionable insights from data"""
    
    def generate_insights(self, df, eda_results):
        """Generate key insights from the analysis"""
        insights = {
            'key_insights': [],
            'trends': [],
            'recommendations': [],
            'anomalies': []
        }
        
        # Insight 1: Missing values
        missing_pct = eda_results['missing_percentages']
        high_missing = [col for col, pct in missing_pct.items() if pct > 20]
        if high_missing:
            insights['key_insights'].append(
                f"⚠️ High missing values in {', '.join(high_missing)} columns (>20%)"
            )
        
        # Insight 2: Highly correlated features
        corr_matrix = eda_results.get('correlation_matrix')
        if corr_matrix is not None:
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr.append(f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]}")
            
            if high_corr:
                insights['key_insights'].append(
                    f"🔗 Strong correlations detected: {', '.join(high_corr[:3])}"
                )
        
        # Insight 3: Skewed distributions
        skewness = eda_results.get('skewness', {})
        skewed_cols = [col for col, skew in skewness.items() if abs(skew) > 1]
        if skewed_cols:
            insights['key_insights'].append(
                f"📊 Highly skewed distributions in: {', '.join(skewed_cols[:3])}"
            )
        
        # Insight 4: Data quality
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Check for outliers
            outlier_count = 0
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outlier_count += (z_scores > 3).sum()
            
            if outlier_count > 0:
                insights['anomalies'].append(
                    f"Found {outlier_count} potential outliers in numeric columns"
                )
        
        # Insight 5: Categorical balance
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            if len(value_counts) > 0:
                majority_pct = (value_counts.iloc[0] / len(df)) * 100
                if majority_pct > 80:
                    insights['key_insights'].append(
                        f"🎯 {col} is dominated by '{value_counts.index[0]}' ({majority_pct:.1f}%)"
                    )
        
        # Generate trends
        for col in numeric_cols[:3]:
            insights['trends'].append(
                f"📈 {col}: range from {df[col].min():.2f} to {df[col].max():.2f}, "
                f"average {df[col].mean():.2f}"
            )
        
        # Generate recommendations
        if len(numeric_cols) > 0:
            insights['recommendations'].append(
                "🎯 Consider normalization/scaling for machine learning models"
            )
        
        if any(missing_pct.values()):
            insights['recommendations'].append(
                "🔧 Address missing values before advanced analysis"
            )
        
        if len(categorical_cols) > 0:
            insights['recommendations'].append(
                "📊 Consider one-hot encoding categorical variables for modeling"
            )
        
        return insights