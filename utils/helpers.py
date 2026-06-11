import pandas as pd
import numpy as np

def make_arrow_compatible(df):
    """Convert DataFrame to Arrow-compatible format"""
    if df is None or df.empty:
        return df
    
    df_copy = df.copy()
    
    for col in df_copy.columns:
        # Convert object dtype to string
        if df_copy[col].dtype == 'object':
            try:
                df_copy[col] = df_copy[col].astype(str)
            except:
                df_copy[col] = df_copy[col].astype(str)
        
        # Convert datetime to string
        elif pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Handle timedelta
        elif pd.api.types.is_timedelta64_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].astype(str)
        
        # Handle categorical
        elif pd.api.types.is_categorical_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].astype(str)
        
        # Handle complex numbers
        elif pd.api.types.is_complex_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].astype(str)
    
    return df_copy

def safe_dataframe_display(df, max_rows=100):
    """Safely display a dataframe in Streamlit"""
    if df is None or df.empty:
        st.write("No data to display")
        return
    
    # Limit rows
    if len(df) > max_rows:
        df = df.head(max_rows)
    
    # Make Arrow compatible
    df_display = make_arrow_compatible(df)
    
    # Display
    st.dataframe(df_display, use_container_width=True)