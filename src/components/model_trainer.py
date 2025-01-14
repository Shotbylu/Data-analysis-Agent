from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from typing import Dict, Any, Tuple
import logging
import joblib
from datetime import datetime

class ModelTrainer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'Random Forest': RandomForestRegressor(random_state=42)
        }
        self.results = {}
        self.best_model = None
        self.best_model_name = None

    def train_models(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """Train multiple models and compare their performance"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Train and evaluate each model
            for name, model in self.models.items():
                self.logger.info(f"Training {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Calculate metrics
                self.results[name] = {
                    'train_r2': r2_score(y_train, train_pred),
                    'test_r2': r2_score(y_test, test_pred),
                    'train_mse': mean_squared_error(y_train, train_pred),
                    'test_mse': mean_squared_error(y_test, test_pred),
                    'train_mae': mean_absolute_error(y_train, train_pred),
                    'test_mae': mean_absolute_error(y_test, test_pred),
                    'model': model
                }

            # Find best model based on test R2 score
            self.best_model_name = max(self.results.items(), 
                                     key=lambda x: x[1]['test_r2'])[0]
            self.best_model = self.results[self.best_model_name]['model']

            return self.results

        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise RuntimeError(f"Error during model training: {str(e)}")

    def save_best_model(self, path: str = None) -> str:
        """Save the best performing model"""
        if self.best_model is None:
            raise ValueError("No best model available. Please train models first.")

        try:
            if path is None:
                path = f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            
            joblib.dump(self.best_model, path)
            self.logger.info(f"Best model saved to {path}")
            return path

        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise RuntimeError(f"Error saving model: {str(e)}")

    def get_feature_importance(self, feature_names: list) -> Dict[str, float]:
        """Get feature importance for the best model if available"""
        if self.best_model is None:
            raise ValueError("No best model available. Please train models first.")

        if not hasattr(self.best_model, 'feature_importances_') and \
           not hasattr(self.best_model, 'coef_'):
            return None

        try:
            if hasattr(self.best_model, 'feature_importances_'):
                importances = self.best_model.feature_importances_
            else:
                importances = np.abs(self.best_model.coef_)

            return dict(zip(feature_names, importances))

        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return None
