import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class PredictiveAgent:
    """Automated predictive modeling"""
    
    def _preprocess_data(self, df, target_col):
        """Preprocess data for ML models - handles datetime columns"""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Make a copy to avoid modifying original
        X = X.copy()
        
        # Handle datetime columns - convert to numeric features
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                # Convert datetime to multiple numeric features
                X[col + '_year'] = X[col].dt.year
                X[col + '_month'] = X[col].dt.month
                X[col + '_day'] = X[col].dt.day
                X[col + '_dayofweek'] = X[col].dt.dayofweek
                # Drop original datetime column
                X = X.drop(columns=[col])
            elif X[col].dtype == 'object':
                # Handle categorical variables
                try:
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
                except:
                    # If LabelEncoder fails, use factorize
                    X[col] = pd.factorize(X[col].astype(str))[0]
        
        # Handle any remaining non-numeric columns
        for col in X.columns:
            if X[col].dtype not in ['int64', 'float64', 'int32', 'float32']:
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    X[col] = X[col].astype('category').cat.codes
        
        # Fill any NaN values that might have been created
        X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        
        return X, y
    
    def train_and_predict(self, df, target_col, problem_type='auto'):
        """Train model and generate predictions"""
        try:
            # Preprocess data
            X, y = self._preprocess_data(df, target_col)
            
            # Check if we have valid data
            if X.shape[1] == 0:
                return {'error': 'No valid features after preprocessing'}
            
            if len(y) < 10:
                return {'error': f'Not enough data: only {len(y)} rows. Need at least 10 rows for training.'}
            
            # Detect problem type
            if problem_type == 'auto':
                if y.dtype == 'object' or len(y.unique()) < 10:
                    problem_type = 'classification'
                else:
                    problem_type = 'regression'
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            results = {}
            
            if problem_type == 'regression':
                # Linear Regression
                lr = LinearRegression()
                lr.fit(X_train, y_train)
                lr_pred = lr.predict(X_test)
                
                # Random Forest
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_train, y_train)
                rf_pred = rf.predict(X_test)
                
                # Calculate metrics
                lr_r2 = r2_score(y_test, lr_pred)
                lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
                rf_r2 = r2_score(y_test, rf_pred)
                rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
                
                results = {
                    'Linear Regression': {'r2': lr_r2, 'rmse': lr_rmse},
                    'Random Forest': {'r2': rf_r2, 'rmse': rf_rmse}
                }
                
                # Find best model
                best_model_name = max(results, key=lambda x: results[x]['r2'])
                best_model = lr if best_model_name == 'Linear Regression' else rf
                final_predictions = best_model.predict(X)
                
                results['best_model'] = best_model_name
                results['r2'] = results[best_model_name]['r2']
                results['rmse'] = results[best_model_name]['rmse']
                
            else:  # classification
                # Random Forest Classifier
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_train, y_train)
                rf_pred = rf.predict(X_test)
                rf_accuracy = accuracy_score(y_test, rf_pred)
                
                results = {
                    'Random Forest': {'accuracy': rf_accuracy}
                }
                
                best_model_name = 'Random Forest'
                best_model = rf
                final_predictions = best_model.predict(X)
                
                results['best_model'] = best_model_name
                results['accuracy'] = rf_accuracy
            
            # Feature importance
            if hasattr(best_model, 'feature_importances_'):
                try:
                    feature_importance = dict(zip(X.columns, best_model.feature_importances_))
                    results['feature_importance'] = dict(sorted(feature_importance.items(), 
                                                                key=lambda x: x[1], 
                                                                reverse=True)[:10])
                except:
                    results['feature_importance'] = {}
            else:
                results['feature_importance'] = {}
            
            # Limit predictions for display
            results['predictions'] = final_predictions[:100] if len(final_predictions) > 100 else final_predictions
            results['actual_values'] = y.values[:100] if len(y) > 100 else y.values
            
            return results
            
        except Exception as e:
            return {
                'error': f"Prediction failed: {str(e)}",
                'best_model': 'None',
                'r2': 0,
                'rmse': 0,
                'accuracy': 0,
                'feature_importance': {},
                'predictions': None,
                'actual_values': None
            }