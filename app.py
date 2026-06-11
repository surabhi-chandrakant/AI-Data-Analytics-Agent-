import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import DataLoader
from agents.data_cleaning_agent import DataCleaningAgent
from agents.eda_agent import EDAAgent
from agents.visualization_agent import VisualizationAgent
from agents.insight_agent import InsightAgent
from agents.report_agent import ReportAgent
from agents.chat_agent import ChatAgent
from agents.predictive_agent import PredictiveAgent

# Page configuration
st.set_page_config(
    page_title="AI Data Analytics Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stAlert {
        border-radius: 10px;
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .insight-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .chat-message-user {
        background-color: #1e88e5;
        color: white;
        padding: 12px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: right;
    }
    .chat-message-assistant {
        background-color: #e0e0e0;
        color: #333;
        padding: 12px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
def init_session_state():
    """Initialize all session state variables"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'chat_agent' not in st.session_state:
        st.session_state.chat_agent = ChatAgent()
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = None

# Call initialization
init_session_state()

class AnalyticsOrchestrator:
    """Main orchestrator for all agents"""
    
    def __init__(self):
        self.cleaning_agent = DataCleaningAgent()
        self.eda_agent = EDAAgent()
        self.viz_agent = VisualizationAgent()
        self.insight_agent = InsightAgent()
        self.report_agent = ReportAgent()
        self.predictive_agent = PredictiveAgent()
    
    def process_dataset(self, df):
        """Run complete analysis pipeline"""
        results = {}
        
        # Step 1: Data Cleaning
        with st.spinner("🧹 Cleaning data..."):
            results['cleaned'], results['cleaning_report'] = self.cleaning_agent.clean(df)
            results['quality_score'] = self.cleaning_agent.assess_quality(results['cleaned'])
        
        # Step 2: EDA
        with st.spinner("📊 Performing EDA..."):
            results['eda'] = self.eda_agent.analyze(results['cleaned'])
        
        # Step 3: Generate Visualizations
        with st.spinner("📈 Creating visualizations..."):
            results['visualizations'] = self.viz_agent.create_dashboard(results['cleaned'])
        
        # Step 4: Generate Insights
        with st.spinner("💡 Generating insights..."):
            results['insights'] = self.insight_agent.generate_insights(
                results['cleaned'], 
                results['eda']
            )
        
        return results

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Data Analytics Agent</h1>
        <p>Your intelligent assistant for automated data analysis, insights, and predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create orchestrator if not exists
    if st.session_state.orchestrator is None:
        st.session_state.orchestrator = AnalyticsOrchestrator()
    
    orchestrator = st.session_state.orchestrator
    
    # Sidebar for data upload
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'json'],
            help="Upload CSV, Excel, or JSON files"
        )
        
        if uploaded_file:
            try:
                loader = DataLoader()
                df = loader.load(uploaded_file)
                st.session_state.data = df
                st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
                
                # Reset processed data when new file is uploaded
                st.session_state.cleaned_data = None
                st.session_state.analysis_results = None
                st.session_state.chat_history = []
                st.session_state.predictions = None
                
                # Show data preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # Main area
    if st.session_state.data is not None:
        # Process data if not already processed
        if st.session_state.cleaned_data is None:
            results = orchestrator.process_dataset(st.session_state.data)
            st.session_state.cleaned_data = results['cleaned']
            st.session_state.analysis_results = results
            # Initialize chat agent with the cleaned data
            st.session_state.chat_agent.initialize(st.session_state.cleaned_data)
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview", "🔍 EDA", "📈 Visualizations", 
            "💡 Insights", "💬 Chat with Data", "🎯 Predictions"
        ])
        
        # Tab 1: Overview
        with tab1:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Rows", f"{len(st.session_state.cleaned_data):,}")
            with col2:
                st.metric("📋 Total Columns", len(st.session_state.cleaned_data.columns))
            with col3:
                missing = st.session_state.cleaned_data.isnull().sum().sum()
                st.metric("⚠️ Missing Values", int(missing))
            with col4:
                quality = st.session_state.analysis_results.get('quality_score', 0)
                st.metric("✅ Data Quality", f"{quality:.1%}")
            
            st.subheader("📋 Dataset Info")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Column Information**")
                col_info = pd.DataFrame({
                    'Column': st.session_state.cleaned_data.columns,
                    'Type': st.session_state.cleaned_data.dtypes.astype(str).values,
                    'Non-Null': st.session_state.cleaned_data.count().values,
                    'Unique': st.session_state.cleaned_data.nunique().values
                })
                st.dataframe(col_info, use_container_width=True)
            
            with col2:
                st.write("**Statistical Summary**")
                st.dataframe(st.session_state.cleaned_data.describe(), use_container_width=True)
        
        # Tab 2: EDA
        with tab2:
            eda_data = st.session_state.analysis_results['eda']
            
            # Correlation heatmap
            st.subheader("🔥 Correlation Matrix")
            numeric_cols = st.session_state.cleaned_data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 1 and eda_data.get('correlation_matrix') is not None:
                fig = px.imshow(
                    eda_data['correlation_matrix'],
                    color_continuous_scale='RdBu',
                    aspect='auto',
                    title="Correlation Heatmap"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for correlation analysis")
            
            # Missing values
            st.subheader("📊 Missing Values Analysis")
            missing_df = pd.DataFrame({
                'Column': list(eda_data.get('missing_counts', {}).keys()),
                'Missing Count': list(eda_data.get('missing_counts', {}).values()),
                'Missing %': list(eda_data.get('missing_percentages', {}).values())
            })
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)
        
        # Tab 3: Visualizations
        with tab3:
            st.subheader("📈 Interactive Visualizations")
            
            numeric_cols = st.session_state.cleaned_data.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = st.session_state.cleaned_data.select_dtypes(include=['object', 'category']).columns.tolist()
            date_cols = st.session_state.cleaned_data.select_dtypes(include=['datetime64']).columns.tolist()
            
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Line Chart"]
            )
            
            if chart_type == "Histogram" and numeric_cols:
                selected_col = st.selectbox("Select Column for Histogram", numeric_cols)
                fig = px.histogram(
                    st.session_state.cleaned_data, 
                    x=selected_col,
                    title=f"Distribution of {selected_col}",
                    color_discrete_sequence=['#667eea'],
                    marginal='box'
                )
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Scatter Plot" and len(numeric_cols) >= 2:
                col1 = st.selectbox("X Axis", numeric_cols, key="scatter_x")
                col2 = st.selectbox("Y Axis", numeric_cols, key="scatter_y")
                fig = px.scatter(
                    st.session_state.cleaned_data,
                    x=col1,
                    y=col2,
                    title=f"{col1} vs {col2}"
                )
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Bar Chart" and categorical_cols:
                cat_col = st.selectbox("Select Category Column", categorical_cols)
                counts = st.session_state.cleaned_data[cat_col].value_counts().head(10)
                fig = px.bar(
                    x=counts.index,
                    y=counts.values,
                    title=f"Top 10 {cat_col}",
                    color_discrete_sequence=['#667eea']
                )
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Box Plot" and numeric_cols:
                fig = px.box(
                    st.session_state.cleaned_data[numeric_cols],
                    title="Box Plots - Outlier Detection"
                )
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Line Chart" and (date_cols or len(numeric_cols) >= 2):
                if date_cols:
                    date_col = st.selectbox("Select Date Column", date_cols)
                    if numeric_cols:
                        value_cols = st.multiselect("Select Value Columns", numeric_cols, default=[numeric_cols[0]] if numeric_cols else [])
                        if value_cols:
                            fig = px.line(
                                st.session_state.cleaned_data,
                                x=date_col,
                                y=value_cols,
                                title=f"Trend Over Time"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                elif len(numeric_cols) >= 2:
                    x_col = st.selectbox("X Axis", numeric_cols, key="line_x")
                    y_col = st.selectbox("Y Axis", [c for c in numeric_cols if c != x_col], key="line_y")
                    fig = px.line(
                        st.session_state.cleaned_data,
                        x=x_col,
                        y=y_col,
                        title=f"{y_col} vs {x_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select appropriate columns for the chosen chart type")
        
        # Tab 4: Insights
        with tab4:
            insights = st.session_state.analysis_results['insights']
            
            st.subheader("🎯 Key Insights")
            for insight in insights.get('key_insights', []):
                st.markdown(f"""
                <div class="insight-card">
                    <strong>💡 Insight:</strong> {insight}
                </div>
                """, unsafe_allow_html=True)
            
            if insights.get('trends'):
                st.subheader("📈 Trend Analysis")
                for trend in insights.get('trends', []):
                    st.info(trend)
            
            if insights.get('recommendations'):
                st.subheader("🎯 Recommendations")
                for rec in insights.get('recommendations', []):
                    st.success(rec)
        
        # Tab 5: Chat with Data - WORKING VERSION
        with tab5:
            st.subheader("💬 Ask questions about your data")
            
            # Create a container for chat history with a fixed height and scroll
            chat_container = st.container()
            
            with chat_container:
                if len(st.session_state.chat_history) == 0:
                    st.info("💡 Ask me anything about your data! Try asking 'How many rows are in the dataset?' or 'Show me median values'")
                else:
                    for msg in st.session_state.chat_history:
                        if msg['role'] == 'user':
                            st.markdown(f"""
                            <div class="chat-message-user">
                                <strong>👤 You:</strong><br>{msg['content']}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="chat-message-assistant">
                                <strong>🤖 AI Agent:</strong><br>{msg['content']}
                            </div>
                            """, unsafe_allow_html=True)
            
            # Separator
            st.markdown("---")
            
            # Input area using columns
            col1, col2 = st.columns([4, 1])
            with col1:
                user_question = st.text_input(
                    "Ask a question:", 
                    key="chat_input_field",
                    placeholder="Example: How many rows are in the dataset? or Show me median values",
                    label_visibility="collapsed"
                )
            with col2:
                send_button = st.button("📤 Send", type="primary", use_container_width=True)
            
            # Process the question
            if send_button and user_question:
                # Add user message
                st.session_state.chat_history.append({'role': 'user', 'content': user_question})
                
                # Get AI response
                with st.spinner("🤔 Analyzing your question..."):
                    response = st.session_state.chat_agent.ask(user_question)
                    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                
                # Rerun to display
                st.rerun()
            
            # Clear chat button
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
            
            # Sample questions
            with st.expander("📝 Sample Questions You Can Ask"):
                st.markdown("""
                **Basic Information:**
                - How many rows are in the dataset?
                - What columns do I have?
                - Show me the dataset summary
                
                **Statistics:**
                - Show me median values
                - Calculate the mean of numeric columns
                - What are the maximum values?
                
                **Data Quality:**
                - Are there any missing values?
                - Show me unique values in categorical columns
                - What is the data quality score?
                """)
        
        # Tab 6: Predictions
        with tab6:
            st.subheader("🎯 Predictive Analytics")
            
            numeric_cols = st.session_state.cleaned_data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                target_col = st.selectbox("Select target column for prediction", numeric_cols)
                
                if st.button("🚀 Train Model & Generate Predictions", type="primary"):
                    with st.spinner("Training machine learning model..."):
                        results = orchestrator.predictive_agent.train_and_predict(
                            st.session_state.cleaned_data,
                            target_col
                        )
                        st.session_state.predictions = results
                    
                    if 'error' in results:
                        st.error(f"Error: {results['error']}")
                    else:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if 'r2' in results:
                                st.metric("Model R² Score", f"{results['r2']:.3f}")
                            elif 'accuracy' in results:
                                st.metric("Model Accuracy", f"{results['accuracy']:.3f}")
                        with col2:
                            if 'rmse' in results:
                                st.metric("Model RMSE", f"{results['rmse']:.3f}")
                        with col3:
                            st.metric("Best Model", results.get('best_model', 'N/A'))
                        
                        if results.get('feature_importance'):
                            st.subheader("📊 Feature Importance")
                            imp_df = pd.DataFrame({
                                'Feature': list(results['feature_importance'].keys()),
                                'Importance': list(results['feature_importance'].values())
                            }).sort_values('Importance', ascending=True)
                            
                            fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                                        title="Top 10 Most Important Features")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        if results.get('predictions') is not None and results.get('actual_values') is not None:
                            st.subheader("📈 Actual vs Predicted Values")
                            n_points = min(100, len(results['predictions']))
                            pred_df = pd.DataFrame({
                                'Actual': results['actual_values'][:n_points],
                                'Predicted': results['predictions'][:n_points]
                            })
                            
                            fig = px.scatter(pred_df, x='Actual', y='Predicted',
                                           title="Actual vs Predicted Values",
                                           opacity=0.6)
                            
                            min_val = min(pred_df['Actual'].min(), pred_df['Predicted'].min())
                            max_val = max(pred_df['Actual'].max(), pred_df['Predicted'].max())
                            fig.add_scatter(x=[min_val, max_val], y=[min_val, max_val],
                                          mode='lines', line=dict(dash='dash', color='red', width=2),
                                          name='Perfect Prediction')
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No numeric columns found for predictive modeling.")
        
        # Download report button
        if st.sidebar.button("📄 Generate Full Report", type="primary"):
            with st.spinner("Generating comprehensive report..."):
                report_path = orchestrator.report_agent.generate_report(
                    st.session_state.cleaned_data,
                    st.session_state.analysis_results
                )
                if os.path.exists(report_path):
                    with open(report_path, 'rb') as f:
                        st.sidebar.download_button(
                            "📥 Download Report",
                            f,
                            file_name="data_analysis_report.pdf",
                            mime="application/pdf"
                        )
                    st.sidebar.success("✅ Report generated successfully!")
        
        # Show cleaning actions in sidebar
        with st.sidebar.expander("🔧 Data Cleaning Actions"):
            cleaning_report = st.session_state.analysis_results.get('cleaning_report', {})
            for action in cleaning_report.get('actions', []):
                st.write(f"✓ {action}")

if __name__ == "__main__":
    main()