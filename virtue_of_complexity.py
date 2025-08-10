import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings

class VirtueOfComplexityModel:
    """
    Implementation of the Virtue of Complexity model from Kelly, Malamud, and Zhou (2022).
    
    This class implements random Fourier features for return prediction with ridge regularization.
    """
    
    def __init__(self, gamma_grid: Optional[list] = None, random_seed: int = 42):
        """
        Initialize the model with Gaussian mixture kernel parameters.
        
        Args:
            gamma_grid: List of gamma values for Gaussian mixture kernel. 
                       Default is [0.1, 0.5, 1, 2, 4, 8, 16] from the paper.
            random_seed: Random seed for reproducibility
        """
        self.gamma_grid = gamma_grid if gamma_grid is not None else [0.1, 0.5, 1, 2, 4, 8, 16]
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        
    def load_data(self, csv_path: str, return_col: str, signal_cols: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from CSV file and extract returns and signals.
        
        Args:
            csv_path: Path to CSV file
            return_col: Column name for returns
            signal_cols: List of column names for signals
            
        Returns:
            Tuple of (returns, signals) arrays
        """
        df = pd.read_csv(csv_path)
        
        if return_col not in df.columns:
            raise ValueError(f"Return column '{return_col}' not found in data")
        
        missing_cols = [col for col in signal_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Signal columns {missing_cols} not found in data")
        
        returns = df[return_col].values
        signals = df[signal_cols].values
        
        # Remove any rows with NaN values
        mask = ~(np.isnan(returns) | np.isnan(signals).any(axis=1))
        returns = returns[mask]
        signals = signals[mask]
        
        return returns, signals
    
    def standardize_signals(self, signals: np.ndarray) -> np.ndarray:
        """
        Standardize signals by their historical standard deviation and clip outliers.
        
        Args:
            signals: Raw signals array
            
        Returns:
            Standardized signals array
        """
        # Fit scaler and transform
        standardized = self.scaler.fit_transform(signals)
        
        # Clip at (-5, 5) as mentioned in the paper
        standardized = np.clip(standardized, -5, 5)
        
        return standardized
    
    def generate_random_features(self, signals: np.ndarray, n_features: int) -> np.ndarray:
        """
        Generate random Fourier features using Gaussian mixture kernel.
        
        Args:
            signals: Standardized input signals [T x d]
            n_features: Number of random features to generate (L)
            
        Returns:
            Random features array [T x L]
        """
        np.random.seed(self.random_seed)
        
        T, d = signals.shape
        p = n_features // len(self.gamma_grid)  # Features per gamma
        
        features_list = []
        
        for gamma in self.gamma_grid:
            # Sample random weights from N(0, I)
            omega = np.random.normal(0, 1, (p, d))
            
            # Compute linear combinations
            linear_combo = signals @ omega.T  # [T x p]
            
            # Apply sin and cos transformations
            sin_features = np.sin(gamma * linear_combo)
            cos_features = np.cos(gamma * linear_combo)
            
            features_list.extend([sin_features, cos_features])
        
        # Concatenate all features
        all_features = np.concatenate(features_list, axis=1)
        
        # Normalize by sqrt(P) as in equation (37)
        P = all_features.shape[1]
        all_features = all_features / np.sqrt(P)
        
        # Take first L features and apply random permutation
        if all_features.shape[1] > n_features:
            perm = np.random.permutation(all_features.shape[1])[:n_features]
            all_features = all_features[:, perm]
        
        return all_features
    
    def ridge_regression(self, X: np.ndarray, y: np.ndarray, ridge_penalty: float) -> np.ndarray:
        """
        Perform ridge regression with given penalty.
        
        Args:
            X: Feature matrix [T x L]
            y: Target returns [T]
            ridge_penalty: Ridge penalty parameter (z)
            
        Returns:
            Regression coefficients [L]
        """
        ridge = Ridge(alpha=ridge_penalty, fit_intercept=False)
        ridge.fit(X, y)
        return ridge.coef_
    
    def rolling_window_prediction(self, returns: np.ndarray, signals: np.ndarray, 
                                window_size: int, n_features: int, 
                                ridge_penalty: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform rolling window out-of-sample prediction.
        
        Args:
            returns: Time series of returns
            signals: Time series of signals
            window_size: Rolling window size (T)
            n_features: Number of random features (L)
            ridge_penalty: Ridge penalty parameter (z)
            
        Returns:
            Tuple of (predictions, actual_returns, coefficients_norm)
        """
        n_obs = len(returns)
        predictions = []
        actual_oos = []
        coef_norms = []
        
        # Start predictions after we have enough data for the window
        for t in range(window_size, n_obs):
            # Training window: [t-window_size : t]
            train_returns = returns[t-window_size:t]
            train_signals = signals[t-window_size:t]
            
            # Standardize signals on training data
            train_signals_std = self.standardize_signals(train_signals)
            
            # Generate random features
            train_features = self.generate_random_features(train_signals_std, n_features)
            
            # Fit ridge regression
            coefficients = self.ridge_regression(train_features, train_returns, ridge_penalty)
            
            # Make prediction for time t
            # Use the same scaler fitted on training data for test signal
            test_signal = signals[t:t+1]  # Keep 2D shape
            test_signal_std = self.scaler.transform(test_signal)
            test_signal_std = np.clip(test_signal_std, -5, 5)
            
            # Generate features for prediction (use same random seed)
            np.random.seed(self.random_seed)
            test_features = self.generate_random_features(test_signal_std, n_features)
            
            prediction = test_features @ coefficients
            
            predictions.append(prediction[0])
            actual_oos.append(returns[t])
            coef_norms.append(np.linalg.norm(coefficients))
        
        return np.array(predictions), np.array(actual_oos), np.array(coef_norms)
    
    def calculate_performance_metrics(self, predictions: np.ndarray, 
                                    actual_returns: np.ndarray) -> dict:
        """
        Calculate performance metrics including R², Sharpe ratio, etc.
        
        Args:
            predictions: Array of return predictions
            actual_returns: Array of actual returns
            
        Returns:
            Dictionary of performance metrics
        """
        # Out-of-sample R²
        ss_res = np.sum((actual_returns - predictions) ** 2)
        ss_tot = np.sum((actual_returns - np.mean(actual_returns)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Timing strategy returns
        timing_returns = predictions * actual_returns
        
        # Strategy performance metrics
        mean_return = np.mean(timing_returns)
        volatility = np.std(timing_returns)
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0
        
        # Mean squared error
        mse = np.mean((actual_returns - predictions) ** 2)
        
        return {
            'r_squared': r_squared,
            'mean_return': mean_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'mse': mse,
            'n_predictions': len(predictions)
        }
    
    def run_analysis(self, returns: np.ndarray, signals: np.ndarray, 
                    window_size: int, n_features: int, 
                    ridge_penalty: float) -> dict:
        """
        Run complete analysis for given parameters.
        
        Args:
            returns: Time series of returns
            signals: Time series of signals  
            window_size: Rolling window size (T)
            n_features: Number of random features (L)
            ridge_penalty: Ridge penalty parameter (z)
            
        Returns:
            Dictionary containing all results
        """
        # Perform rolling window prediction
        predictions, actual_oos, coef_norms = self.rolling_window_prediction(
            returns, signals, window_size, n_features, ridge_penalty
        )
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(predictions, actual_oos)
        
        # Add coefficient norm statistics
        metrics['mean_coef_norm'] = np.mean(coef_norms)
        metrics['complexity_ratio'] = n_features / window_size
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'actual_returns': actual_oos,
            'coefficient_norms': coef_norms,
            'parameters': {
                'window_size': window_size,
                'n_features': n_features,
                'ridge_penalty': ridge_penalty,
                'random_seed': self.random_seed
            }
        }


def main_analysis(csv_path: str, return_col: str, signal_cols: list,
                 window_size: int = 12, n_features: int = 1000, 
                 ridge_penalty: float = 1.0, random_seed: int = 42) -> dict:
    """
    Main function to run the virtue of complexity analysis.
    
    Args:
        csv_path: Path to CSV file
        return_col: Column name for returns
        signal_cols: List of column names for signals
        window_size: Rolling window size (T)
        n_features: Number of random features (L)  
        ridge_penalty: Ridge penalty parameter (z)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing all analysis results
    """
    # Initialize model
    model = VirtueOfComplexityModel(random_seed=random_seed)
    
    # Load data
    returns, signals = model.load_data(csv_path, return_col, signal_cols)
    
    print(f"Loaded {len(returns)} observations with {signals.shape[1]} signals")
    print(f"Parameters: T={window_size}, L={n_features}, z={ridge_penalty}")
    
    # Run analysis
    results = model.run_analysis(returns, signals, window_size, n_features, ridge_penalty)
    
    # Print summary results
    metrics = results['metrics']
    print("\n" + "="*50)
    print("PERFORMANCE METRICS")
    print("="*50)
    print(f"Out-of-sample R²: {metrics['r_squared']:.4f}")
    print(f"Mean return: {metrics['mean_return']:.4f}")
    print(f"Volatility: {metrics['volatility']:.4f}")
    print(f"Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Mean coefficient norm: {metrics['mean_coef_norm']:.4f}")
    print(f"Complexity ratio (L/T): {metrics['complexity_ratio']:.2f}")
    print(f"Number of predictions: {metrics['n_predictions']}")
    
    return results


if __name__ == "__main__":
    # Example usage
    # You would replace these with your actual file path and column names
    csv_path = "sample_data.csv"
    return_col = "returns"
    signal_cols = ["signal1", "signal2", "signal3"]
    
    try:
        results = main_analysis(
            csv_path=csv_path,
            return_col=return_col, 
            signal_cols=signal_cols,
            window_size=12,
            n_features=1000,
            ridge_penalty=1.0,
            random_seed=42
        )
    except FileNotFoundError:
        print(f"File {csv_path} not found. Please provide a valid CSV file path.")
    except Exception as e:
        print(f"Error: {e}")
