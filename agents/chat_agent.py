import pandas as pd
import numpy as np

class ChatAgent:
    """Natural language query agent for data"""
    
    def __init__(self):
        self.df = None
        
    def initialize(self, df):
        """Initialize the chat agent with dataframe"""
        self.df = df
    
    def _get_numeric_cols(self):
        """Get numeric columns safely"""
        return self.df.select_dtypes(include=[np.number]).columns.tolist()
    
    def _get_categorical_cols(self):
        """Get categorical columns safely"""
        return [col for col in self.df.select_dtypes(include=['object']).columns 
                if not pd.api.types.is_datetime64_any_dtype(self.df[col])]
    
    def ask(self, question):
        """Answer questions about the data"""
        if self.df is None:
            return "⚠️ Please upload a dataset first before asking questions."
        
        if not question or question.strip() == "":
            return "❓ Please enter a question about your data."
        
        # Get response
        response = self._rule_based_response(question)
        
        # Ensure we always return something
        if not response or response.strip() == "":
            return "🤔 I couldn't generate a response. Please try asking a different question."
        
        return response
    
    def _rule_based_response(self, question):
        """Rule-based responses for common questions"""
        question_lower = question.lower().strip()
        
        # Greeting responses
        if any(word in question_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return "👋 Hello! I'm your AI Data Analytics Agent. I can help you analyze your dataset. Try asking me about rows, columns, statistics, or data quality!"
        
        # Help response
        if any(word in question_lower for word in ['help', 'what can you do', 'capabilities']):
            return self._get_help_message()
        
        # Rows/records count
        if any(word in question_lower for word in ['row', 'record', 'how many row', 'number of row', 'count of row']):
            return f"📊 The dataset contains **{len(self.df):,} rows** (records)."
        
        # Columns count and list
        if 'column' in question_lower:
            if 'how many' in question_lower or 'number of' in question_lower:
                return f"📋 The dataset has **{len(self.df.columns)} columns**."
            elif 'list' in question_lower or 'what are' in question_lower or 'show me' in question_lower:
                cols_list = []
                for col in self.df.columns:
                    dtype = str(self.df[col].dtype)
                    if 'datetime' in dtype:
                        dtype = '📅 datetime'
                    elif 'int' in dtype:
                        dtype = '🔢 integer'
                    elif 'float' in dtype:
                        dtype = '📊 float'
                    else:
                        dtype = '📝 text'
                    cols_list.append(f"  • **{col}** ({dtype})")
                
                cols_text = '\n'.join(cols_list[:15])
                if len(self.df.columns) > 15:
                    cols_text += f"\n  • ... and {len(self.df.columns) - 15} more columns"
                
                return f"📋 **Columns in your dataset ({len(self.df.columns)} total):**\n\n{cols_text}"
            else:
                return f"The dataset has **{len(self.df.columns)} columns**. Ask me to 'list all columns' to see them."
        
        # Missing/null values
        if any(word in question_lower for word in ['missing', 'null', 'na', 'empty']):
            missing = self.df.isnull().sum()
            missing_cols = missing[missing > 0]
            if len(missing_cols) > 0:
                total_missing = missing_cols.sum()
                result = f"⚠️ **Missing Values Summary:**\n\n"
                result += f"Total missing values: **{total_missing:,}** out of {len(self.df) * len(self.df.columns):,} cells ({total_missing/(len(self.df) * len(self.df.columns))*100:.1f}%)\n\n"
                result += "**Columns with missing values:**\n"
                for col, count in missing_cols.items():
                    pct = (count / len(self.df)) * 100
                    result += f"  • {col}: {count:,} missing ({pct:.1f}%)\n"
                return result
            else:
                return "✅ **Great news!** No missing values found in the dataset. The data is complete!"
        
        # Mean/average
        if any(word in question_lower for word in ['mean', 'average', 'avg']):
            numeric_cols = self._get_numeric_cols()
            if len(numeric_cols) > 0:
                means = self.df[numeric_cols].mean()
                result = "📈 **Mean (Average) Values:**\n\n"
                for col, val in means.items():
                    result += f"  • {col}: **{val:,.2f}**\n"
                return result
            else:
                return "⚠️ No numeric columns found in the dataset to calculate mean values."
        
        # Median
        if 'median' in question_lower:
            numeric_cols = self._get_numeric_cols()
            if len(numeric_cols) > 0:
                medians = self.df[numeric_cols].median()
                result = "📊 **Median Values:**\n\n"
                for col, val in medians.items():
                    result += f"  • {col}: **{val:,.2f}**\n"
                return result
            else:
                return "⚠️ No numeric columns found in the dataset to calculate median values."
        
        # Max
        if 'max' in question_lower and 'maximum' in question_lower or ('max' in question_lower and not 'min' in question_lower):
            numeric_cols = self._get_numeric_cols()
            if len(numeric_cols) > 0:
                max_vals = self.df[numeric_cols].max()
                result = "🔝 **Maximum Values:**\n\n"
                for col, val in max_vals.items():
                    result += f"  • {col}: **{val:,.2f}**\n"
                return result
            else:
                return "⚠️ No numeric columns found to calculate maximum values."
        
        # Min
        if 'min' in question_lower and 'minimum' in question_lower or ('min' in question_lower and not 'max' in question_lower):
            numeric_cols = self._get_numeric_cols()
            if len(numeric_cols) > 0:
                min_vals = self.df[numeric_cols].min()
                result = "⬇️ **Minimum Values:**\n\n"
                for col, val in min_vals.items():
                    result += f"  • {col}: **{val:,.2f}**\n"
                return result
            else:
                return "⚠️ No numeric columns found to calculate minimum values."
        
        # Correlation
        if 'correlation' in question_lower:
            numeric_cols = self._get_numeric_cols()
            if len(numeric_cols) >= 2:
                corr_matrix = self.df[numeric_cols].corr()
                result = "📉 **Correlation Analysis:**\n\n"
                result += "Strongest correlations (|r| > 0.5):\n"
                correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.5:
                            strength = "strong positive" if corr_val > 0.7 else "moderate positive" if corr_val > 0 else "moderate negative" if corr_val < -0.3 else "weak"
                            correlations.append(f"  • **{corr_matrix.columns[i]}** & **{corr_matrix.columns[j]}**: {corr_val:.3f} ({strength})")
                
                if correlations:
                    result += '\n'.join(correlations[:5])
                else:
                    result += "  No strong correlations found (|r| > 0.5)."
                return result
            else:
                return f"⚠️ Need at least 2 numeric columns for correlation analysis. Found {len(numeric_cols)} numeric column(s)."
        
        # Summary / Describe
        if any(word in question_lower for word in ['summary', 'describe', 'overview', 'tell me about']):
            numeric_cols = self._get_numeric_cols()
            result = f"📊 **Dataset Summary**\n\n"
            result += f"• **Shape**: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns\n"
            result += f"• **Memory**: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n"
            result += f"• **Missing Values**: {self.df.isnull().sum().sum():,}\n"
            result += f"• **Duplicates**: {self.df.duplicated().sum():,}\n\n"
            
            result += "**Column Types:**\n"
            for dtype, count in self.df.dtypes.value_counts().items():
                result += f"  • {dtype}: {count} columns\n"
            
            if len(numeric_cols) > 0:
                result += f"\n**Numeric Statistics (top 5 columns):**\n"
                stats = self.df[numeric_cols[:5]].describe()
                result += str(stats.round(2))
            
            return result
        
        # Unique values
        if any(word in question_lower for word in ['unique', 'distinct']):
            categorical_cols = self._get_categorical_cols()
            if len(categorical_cols) > 0:
                result = "🔢 **Unique Values in Categorical Columns:**\n\n"
                for col in categorical_cols[:10]:
                    unique_count = self.df[col].nunique()
                    result += f"  • {col}: **{unique_count:,}** unique value(s)\n"
                return result
            else:
                return "No categorical (text) columns found in the dataset."
        
        # Data types
        if any(word in question_lower for word in ['type', 'dtype', 'data type']):
            result = "📋 **Column Data Types:**\n\n"
            for col in self.df.columns:
                dtype = str(self.df[col].dtype)
                if 'datetime' in dtype:
                    dtype = '📅 datetime'
                elif 'int' in dtype:
                    dtype = '🔢 integer'
                elif 'float' in dtype:
                    dtype = '📊 float'
                else:
                    dtype = '📝 text/string'
                result += f"  • **{col}**: {dtype}\n"
            return result
        
        # Quality score
        if any(word in question_lower for word in ['quality', 'clean']):
            missing_count = self.df.isnull().sum().sum()
            duplicate_count = self.df.duplicated().sum()
            total_cells = len(self.df) * len(self.df.columns)
            completeness = (1 - missing_count / total_cells) * 100 if total_cells > 0 else 0
            
            result = f"✅ **Data Quality Report:**\n\n"
            result += f"• **Completeness**: {completeness:.1f}% (no missing values in {completeness:.1f}% of cells)\n"
            result += f"• **Duplicate Rows**: {duplicate_count}\n"
            result += f"• **Missing Values**: {missing_count}\n"
            
            if completeness > 95 and duplicate_count == 0:
                result += "\n🎉 **Excellent!** Your data is very clean and ready for analysis."
            elif completeness > 80:
                result += "\n👍 **Good!** Your data has minor issues that can be easily fixed."
            else:
                result += "\n⚠️ **Needs Improvement** - Consider cleaning missing values before analysis."
            
            return result
        
        # Default response if no pattern matches
        return self._get_help_message()
    
    def _get_help_message(self):
        """Get help message with available commands"""
        return """🤖 **I can help you analyze your data!** 

Try asking me things like:

**📊 Basic Info:**
• "How many rows are in the dataset?"
• "What columns do I have?"
• "Show me the dataset summary"

**📈 Statistics:**
• "Calculate the mean of numeric columns"
• "Show me median values"
• "What are the maximum values?"

**🔍 Data Quality:**
• "Are there any missing values?"
• "Show me unique values"
• "What is the data quality score?"

**📋 Other:**
• "List all column names"
• "What are the data types?"
• "Show correlation between columns"

Just type your question and I'll answer! 💬"""