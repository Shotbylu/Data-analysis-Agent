import pandas as pd
import logging
from typing import Optional, Tuple
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data: Optional[pd.DataFrame] = None
        self.target_column: Optional[str] = None

    def load_data(self, file) -> Optional[pd.DataFrame]:
        """Load data from uploaded file"""
        try:
            self.data = pd.read_csv(file)
            self.logger.info(f"Successfully loaded data with shape {self.data.shape}")
            return self.data
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise RuntimeError(f"Error loading file: {str(e)}")

    def prepare_data(self, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        self.target_column = target_col
        df = self.data.copy()

        # Handle missing values
        for column in df.columns:
            if df[column].dtype in [np.number]:
                df[column].fillna(df[column].mean(), inplace=True)
            else:
                df[column].fillna(df[column].mode()[0], inplace=True)

        # Encode categorical variables
        le = LabelEncoder()
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = le.fit_transform(df[column].astype(str))

        # Prepare features and target
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values

        return X, y

    def get_numeric_columns(self) -> list:
        """Get list of numeric columns"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        return self.data.select_dtypes(include=[np.number]).columns.tolist()

    def get_summary_stats(self) -> dict:
        """Get basic summary statistics of the data"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        return {
            "rows": self.data.shape[0],
            "columns": self.data.shape[1],
            "missing_values": self.data.isna().sum().sum(),
            "numeric_columns": len(self.get_numeric_columns()),
            "categorical_columns": len(self.data.select_dtypes(include=['object']).columns)
        }
