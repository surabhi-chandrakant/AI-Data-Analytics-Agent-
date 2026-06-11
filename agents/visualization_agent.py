import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class VisualizationAgent:
    """Automated chart generation"""
    
    def create_dashboard(self, df):
        """Create a comprehensive dashboard of visualizations"""
        charts = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # 1. Distribution plots for numeric columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:3]:  # Limit to first 3 for performance
                fig = px.histogram(
                    df, x=col, 
                    title=f"Distribution of {col}",
                    color_discrete_sequence=['#667eea'],
                    marginal='box'
                )
                charts[f'{col}_distribution'] = fig
        
        # 2. Box plots for outlier detection
        if len(numeric_cols) > 0:
            fig = px.box(
                df[numeric_cols], 
                title="Box Plots - Outlier Detection",
                color_discrete_sequence=['#764ba2']
            )
            charts['box_plots'] = fig
        
        # 3. Correlation heatmap
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu',
                title="Correlation Heatmap"
            )
            charts['correlation_heatmap'] = fig
        
        # 4. Bar charts for categorical columns
        if len(categorical_cols) > 0:
            for col in categorical_cols[:2]:  # Limit to first 2
                value_counts = df[col].value_counts().head(10)
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Top 10 {col}",
                    color_discrete_sequence=['#667eea'],
                    labels={'x': col, 'y': 'Count'}
                )
                charts[f'{col}_bar_chart'] = fig
        
        # 5. Scatter matrix for relationships (if not too many columns)
        if 2 <= len(numeric_cols) <= 5:
            fig = px.scatter_matrix(
                df[numeric_cols],
                title="Scatter Matrix",
                color_discrete_sequence=['#667eea']
            )
            charts['scatter_matrix'] = fig
        
        return charts