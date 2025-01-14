import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import numpy as np

class DataAnalyzer:
    def __init__(self, data: Optional[pd.DataFrame] = None):
        self.data = data

    def set_data(self, data: pd.DataFrame):
        """Set the data to analyze"""
        self.data = data

    def get_missing_values_analysis(self) -> Dict[str, Any]:
        """Analyze missing values in the dataset"""
        if self.data is None:
            raise ValueError("No data set for analysis")

        missing_data = self.data.isnull().sum().reset_index()
        missing_data.columns = ['Column', 'Missing Count']
        missing_data['Missing Percentage'] = (missing_data['Missing Count'] / len(self.data)) * 100

        fig = px.bar(
            missing_data,
            x='Column',
            y='Missing Percentage',
            title='Missing Values by Column'
        )

        return {
            'data': missing_data,
            'plot': fig
        }

    def get_distribution_plot(self, column: str):
        """Generate distribution plot for a specific column"""
        if self.data is None:
            raise ValueError("No data set for analysis")

        if column not in self.data.columns:
            raise ValueError(f"Column {column} not found in dataset")

        fig = px.histogram(
            self.data,
            x=column,
            title=f'Distribution of {column}'
        )

        return fig

    def get_correlation_matrix(self):
        """Generate correlation matrix for numeric columns"""
        if self.data is None:
            raise ValueError("No data set for analysis")

        numeric_df = self.data.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("No numeric columns found in dataset")

        corr_matrix = numeric_df.corr()
        fig = px.imshow(
            corr_matrix,
            title='Correlation Matrix',
            color_continuous_scale='RdBu'
        )

        return fig

    def get_basic_stats(self) -> pd.DataFrame:
        """Get basic statistical description of the data"""
        if self.data is None:
            raise ValueError("No data set for analysis")

        return self.data.describe()

    def get_column_types(self) -> Dict[str, list]:
        """Get columns categorized by their types"""
        if self.data is None:
            raise ValueError("No data set for analysis")

        return {
            'numeric': self.data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical': self.data.select_dtypes(include=['object']).columns.tolist(),
            'datetime': self.data.select_dtypes(include=['datetime64']).columns.tolist(),
            'boolean': self.data.select_dtypes(include=['bool']).columns.tolist()
        }
