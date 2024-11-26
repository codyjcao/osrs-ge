from statsmodels.tsa.arima.model import ARIMA
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import numpy as np
import pandas as pd

class ARIMAHybridModel(BaseEstimator, RegressorMixin):
    def __init__(self, arima_order=(1, 1, 1), ml_model=None, ml_lag_features=4, decay_rate=0.95):
        """
        Initializes the ARIMAHybridModel with support for exponential sample weighting.

        Parameters:
        - arima_order: Tuple (p, d, q) for the ARIMA model.
        - ml_model: Machine learning model that follows scikit-learn's interface.
        - ml_lag_features: Number of lagged features to create for the ML model.
        - decay_rate: Decay rate for exponential weights, only used if sample_weight is not provided in fit.
        """
        self.arima_order = arima_order
        self.ml_model = ml_model
        self.ml_lag_features = ml_lag_features
        self.decay_rate = decay_rate
        self.arima_model = None
        self.ml_features_ = None
    
    def fit(self, X, y, sample_weight=None):
        """
        Fits the ARIMA and ML models with optional sample weighting.
        
        Parameters:
        - X: Feature data, typically a DataFrame with price/volume data.
        - y: Target variable, typically a Series of prices.
        - sample_weight: Optional array-like custom sample weights. If None, uses exponential decay.
        """
        # Fit ARIMA on the target variable y
        self.arima_model = ARIMA(y, order=self.arima_order).fit()
        
        # Calculate ARIMA residuals
        arima_predictions = self.arima_model.fittedvalues
        residuals = y - arima_predictions
        
        # Generate lagged features from residuals for the ML model
        self.ml_features_ = self._create_lagged_features(residuals)
        
        # Generate exponentially decaying sample weights if none are provided
        if sample_weight is None:
            sample_weight = self._generate_exponential_weights(len(self.ml_features_))
        
        # Ensure sample_weight matches the index of self.ml_features_
        sample_weight = sample_weight[self.ml_features_.index]
        
        # Fit the ML model on lagged features and residuals with sample weights
        if self.ml_model:
            self.ml_model.fit(self.ml_features_, residuals[self.ml_features_.index], sample_weight=sample_weight)
        
        return self

    def predict(self, X):
        """
        Predicts the target variable using the hybrid model.
        
        Parameters:
        - X: Feature data, typically a DataFrame with price/volume data.
        
        Returns:
        - Predictions of the target variable.
        """
        # Get ARIMA forecast
        arima_forecast = self.arima_model.forecast(steps=len(X))
        
        # Generate lagged features for the ML model based on forecast residuals
        ml_features = self._create_lagged_features(self.arima_model.resid, forecast_mode=True, num_steps=len(X))
        
        # ML model residual predictions, if ML model was provided
        if self.ml_model:
            ml_forecast = self.ml_model.predict(ml_features)
        else:
            ml_forecast = np.zeros(len(X))  # If no ML model, no residual adjustment
        
        # Final forecast by combining ARIMA and ML model predictions
        final_forecast = arima_forecast + ml_forecast
        return final_forecast

    def _create_lagged_features(self, data, forecast_mode=False, num_steps=0):
        """
        Creates lagged features for the ML model.
        
        Parameters:
        - data: Residual data from ARIMA model for lagged feature generation.
        - forecast_mode: Whether to generate features for forecast period (True) or training period (False).
        - num_steps: Number of steps to forecast ahead, only used if forecast_mode is True.
        
        Returns:
        - DataFrame with lagged features.
        """
        lagged_features = pd.DataFrame()
        
        # Generate lagged features
        for lag in range(1, self.ml_lag_features + 1):
            if forecast_mode:
                lagged_features[f'lag_{lag}'] = data[-num_steps - lag: -lag].values
            else:
                lagged_features[f'lag_{lag}'] = data.shift(lag)
        
        return lagged_features.dropna() if not forecast_mode else lagged_features

    def _generate_exponential_weights(self, n):
        """
        Generates an array of exponentially decaying weights.

        Parameters:
        - n: The length of the weights array to generate.

        Returns:
        - Array of weights with exponential decay.
        """
        return np.array([self.decay_rate ** i for i in range(n)][::-1])  # Highest weight for the most recent





class ARIMAHybridClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, arima_order=(1, 1, 1), ml_model=None, ml_lag_features=4, 
                 pos_threshold=0.01, neg_threshold=-0.01, decay_rate=0.95):
        """
        Initializes the ARIMAHybridClassifier with support for additional non-lagged features.

        Parameters:
        - arima_order: Tuple (p, d, q) for the ARIMA model.
        - ml_model: Machine learning model that follows scikit-learn's interface for classification.
        - ml_lag_features: Number of lagged features to create for the ML model.
        - pos_threshold: Positive threshold for classifying returns as 1.
        - neg_threshold: Negative threshold for classifying returns as -1.
        - decay_rate: Decay rate for exponential weights, only used if sample_weight is not provided in fit.
        """
        self.arima_order = arima_order
        self.ml_model = ml_model
        self.ml_lag_features = ml_lag_features
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.decay_rate = decay_rate
        self.arima_model = None
        self.ml_features_ = None

    def fit(self, X, y, sample_weight=None):
        """
        Fits the ARIMA and ML models with optional sample weighting.
    
        Parameters:
        - X: Feature data (e.g., price, volume) to include in the ML model.
        - y: Target variable (e.g., price or returns).
        - sample_weight: Optional array-like custom sample weights. If None, uses exponential decay.
        """
        # Fit ARIMA on the target variable y
        self.arima_model = ARIMA(y, order=self.arima_order).fit()
        
        # Calculate ARIMA residuals
        arima_predictions = self.arima_model.fittedvalues
        residuals = y - arima_predictions
        
        # Classify the actual returns based on thresholds
        classified_returns = self._classify_returns(y)
        
        # Generate lagged features from ARIMA residuals
        lagged_residual_features = self._create_lagged_features(residuals)
        
        # Combine lagged residuals and non-lagged features
        self.ml_features_ = pd.concat([lagged_residual_features, X.loc[lagged_residual_features.index]], axis=1)
        
        # Generate exponentially decaying sample weights if none are provided
        if sample_weight is None:
            sample_weight = self._generate_exponential_weights(len(self.ml_features_))
        

        # Fit the ML model on combined features with classified returns as the target
        if self.ml_model is not None:
            self.ml_model.fit(self.ml_features_, classified_returns[self.ml_features_.index], sample_weight=sample_weight)
        
        return self


    def predict(self, X, steps=None):
        """
        Predicts the target class using the hybrid model.
    
        Parameters:
        - X: Feature data (e.g., price, volume). Can be a DataFrame (multiple rows) or a Series (single row).
        - steps: Optional, number of future steps to predict. Defaults to len(X) if X is a DataFrame.
    
        Returns:
        - Predictions of the target classes.
            - A list of predictions if X is a DataFrame.
            - A single prediction if X is a Series.
        """
        # Determine the forecast horizon
        if isinstance(X, pd.DataFrame):
            # If X is a DataFrame, each row corresponds to one prediction
            if steps is None:
                steps = len(X)
        elif isinstance(X, pd.Series):
            # If X is a Series, we assume a single prediction is requested
            steps = 1
            X = X.to_frame().T  # Convert Series to a single-row DataFrame for consistency
        else:
            raise ValueError("Input X must be a pandas DataFrame or Series.")
    
        # Get ARIMA forecast
        arima_forecast = self.arima_model.forecast(steps=steps)
    
        # Convert ARIMA forecast to three classes based on thresholds
        arima_classes = self._classify_returns(arima_forecast)
    
        # Generate lagged features for the ML model based on forecast residuals
        lagged_residual_features = self._create_lagged_features(self.arima_model.resid, forecast_mode=True, num_steps=steps)
    
        # Combine lagged residuals and non-lagged features
        combined_features = pd.concat([lagged_residual_features, X.iloc[:steps].reset_index(drop=True)], axis=1)
    
        # ML model class predictions, if ML model was provided
        if self.ml_model:
            ml_forecast = self.ml_model.predict(combined_features)
        else:
            ml_forecast = arima_classes  # If no ML model, fall back on ARIMA classes
    
        # Return a single prediction if input was a Series
        if isinstance(X, pd.DataFrame) and len(X) == 1:
            return ml_forecast[0]  # Return a single number
        return ml_forecast  # Return a list for multiple rows

    def _create_lagged_features(self, data, forecast_mode=False, num_steps=0):
        """
        Creates lagged features for the ML model.
    
        Parameters:
        - data: Residual data from ARIMA model for lagged feature generation.
        - forecast_mode: Whether to generate features for forecast period (True) or training period (False).
        - num_steps: Number of steps to forecast ahead, only used if forecast_mode is True.
    
        Returns:
        - DataFrame with lagged features, indexed to match the input data.
        """
        lagged_features = pd.DataFrame()
    
        for lag in range(1, self.ml_lag_features + 1):
            if forecast_mode:
                # Adjust slicing to ensure proper alignment
                lagged_values = data[-num_steps - lag: -lag]
                if len(lagged_values) != num_steps:  # Ensure we have the right number of values
                    lagged_values = lagged_values[-num_steps:]  # Truncate to match `num_steps`
                # Preserve index
                lagged_features[f'lag_{lag}'] = lagged_values.values
                lagged_features.index = data.index[-num_steps:]  # Align the index with the forecast period
            else:
                # Preserve index by using the shift method
                lagged_features[f'lag_{lag}'] = data.shift(lag)
    
        # Drop NaN values for training (forecast_mode=False)
        return lagged_features.dropna() if not forecast_mode else lagged_features



    def _classify_returns(self, returns):
        """
        Classifies returns into -1, 0, or 1 based on positive and negative thresholds.
    
        Parameters:
        - returns: Series or array-like of returns to classify.
    
        Returns:
        - Series of classified returns (indexed the same as the input).
        """
        # Ensure input is a pandas Series
        returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
    
        # Apply classification logic while preserving the index
        classified_returns = pd.Series(
            np.where(returns > self.pos_threshold, 1,
                     np.where(returns < self.neg_threshold, -1, 0)),
            index=returns.index
        )
        return classified_returns
    
    def _generate_exponential_weights(self, n):
        """
        Generates an array of exponentially decaying weights.

        Parameters:
        - n: The length of the weights array to generate.

        Returns:
        - Array of weights with exponential decay.
        """
        return np.array([self.decay_rate ** i for i in range(n)][::-1])  # Highest weight for the most recent

