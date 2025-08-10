import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from io import StringIO
import sys

# Import the main module (assuming it's saved as virtue_of_complexity.py)
try:
    from virtue_of_complexity import VirtueOfComplexityModel, main_analysis
except ImportError as e:
    print(f"Please ensure the virtue_of_complexity.py file is in the same directory {e}")
    sys.exit(1)


class TestVirtueOfComplexityModel(unittest.TestCase):
    """Unit tests for the VirtueOfComplexityModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = VirtueOfComplexityModel(random_seed=42)
        
        # Create sample data
        np.random.seed(42)
        self.n_obs = 100
        self.n_signals = 3
        
        self.sample_returns = np.random.normal(0.01, 0.2, self.n_obs)
        self.sample_signals = np.random.normal(0, 1, (self.n_obs, self.n_signals))
        
        # Create temporary CSV file for testing
        self.temp_csv = self._create_temp_csv()
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_csv):
            os.unlink(self.temp_csv)
    
    def _create_temp_csv(self):
        """Create a temporary CSV file for testing."""
        df = pd.DataFrame({
            'returns': self.sample_returns,
            'signal1': self.sample_signals[:, 0],
            'signal2': self.sample_signals[:, 1], 
            'signal3': self.sample_signals[:, 2]
        })
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        return temp_file.name
    
    def test_initialization(self):
        """Test model initialization."""
        # Test default initialization
        model1 = VirtueOfComplexityModel()
        self.assertEqual(model1.gamma_grid, [0.1, 0.5, 1, 2, 4, 8, 16])
        self.assertEqual(model1.random_seed, 42)
        
        # Test custom initialization
        custom_gamma = [0.5, 1.0, 2.0]
        model2 = VirtueOfComplexityModel(gamma_grid=custom_gamma, random_seed=123)
        self.assertEqual(model2.gamma_grid, custom_gamma)
        self.assertEqual(model2.random_seed, 123)
    
    def test_load_data(self):
        """Test data loading functionality."""
        # Test successful loading
        returns, signals = self.model.load_data(
            self.temp_csv, 'returns', ['signal1', 'signal2', 'signal3']
        )
        
        self.assertEqual(len(returns), self.n_obs)
        self.assertEqual(signals.shape, (self.n_obs, self.n_signals))
        self.assertTrue(isinstance(returns, np.ndarray))
        self.assertTrue(isinstance(signals, np.ndarray))
        
        # Test missing return column
        with self.assertRaises(ValueError):
            self.model.load_data(self.temp_csv, 'missing_return', ['signal1'])
        
        # Test missing signal column
        with self.assertRaises(ValueError):
            self.model.load_data(self.temp_csv, 'returns', ['missing_signal'])
    
    def test_load_data_with_nans(self):
        """Test data loading with NaN values."""
        # Create data with NaN values
        df_with_nans = pd.DataFrame({
            'returns': [0.01, np.nan, 0.02, 0.03],
            'signal1': [1.0, 2.0, np.nan, 4.0],
            'signal2': [0.5, 1.5, 2.5, 3.5]
        })
        
        temp_file_nans = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df_with_nans.to_csv(temp_file_nans.name, index=False)
        temp_file_nans.close()
        
        try:
            returns, signals = self.model.load_data(
                temp_file_nans.name, 'returns', ['signal1', 'signal2']
            )
            
            # Should only have 1 valid row (index 3)
            self.assertEqual(len(returns), 1)
            self.assertEqual(signals.shape, (1, 2))
            self.assertAlmostEqual(returns[0], 0.03, places=6)
            
        finally:
            os.unlink(temp_file_nans.name)
    
    def test_standardize_signals(self):
        """Test signal standardization."""
        signals = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        
        standardized = self.model.standardize_signals(signals)
        
        # Check shape preservation
        self.assertEqual(standardized.shape, signals.shape)
        
        # Check that each column is approximately standardized
        for i in range(signals.shape[1]):
            self.assertAlmostEqual(np.mean(standardized[:, i]), 0, places=10)
            self.assertAlmostEqual(np.std(standardized[:, i], ddof=0), 1, places=10)
        
        # Check clipping
        extreme_signals = np.array([[100, -100, 0]])
        clipped = self.model.standardize_signals(extreme_signals)
        self.assertTrue(np.all(clipped >= -5))
        self.assertTrue(np.all(clipped <= 5))
    
    def test_generate_random_features(self):
        """Test random feature generation."""
        signals = np.random.normal(0, 1, (50, 3))
        n_features = 100
        
        features = self.model.generate_random_features(signals, n_features)
        
        # Check output dimensions
        self.assertEqual(features.shape[0], signals.shape[0])
        self.assertLessEqual(features.shape[1], n_features)
        
        # Check that features are finite
        self.assertTrue(np.all(np.isfinite(features)))
        
        # Test reproducibility with same seed
        features2 = self.model.generate_random_features(signals, n_features)
        np.testing.assert_array_equal(features, features2)
        
        # Test different results with different seeds
        model_diff_seed = VirtueOfComplexityModel(random_seed=123)
        features3 = model_diff_seed.generate_random_features(signals, n_features)
        self.assertFalse(np.array_equal(features, features3))
    
    def test_ridge_regression(self):
        """Test ridge regression functionality."""
        # Create simple test case
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        y = np.array([1, 2, 3], dtype=float)
        ridge_penalty = 0.1
        
        coefficients = self.model.ridge_regression(X, y, ridge_penalty)
        
        # Check output shape
        self.assertEqual(len(coefficients), X.shape[1])
        
        # Check that coefficients are finite
        self.assertTrue(np.all(np.isfinite(coefficients)))
        
        # Test with different penalty values
        coef_small_penalty = self.model.ridge_regression(X, y, 0.01)
        coef_large_penalty = self.model.ridge_regression(X, y, 10.0)
        
        # Larger penalty should result in smaller coefficient norms
        self.assertGreater(np.linalg.norm(coef_small_penalty), 
                          np.linalg.norm(coef_large_penalty))
    
    def test_rolling_window_prediction(self):
        """Test rolling window prediction."""
        returns = self.sample_returns
        signals = self.sample_signals
        window_size = 20
        n_features = 50
        ridge_penalty = 1.0
        
        predictions, actual_oos, coef_norms = self.model.rolling_window_prediction(
            returns, signals, window_size, n_features, ridge_penalty
        )
        
        # Check output dimensions
        expected_predictions = len(returns) - window_size
        self.assertEqual(len(predictions), expected_predictions)
        self.assertEqual(len(actual_oos), expected_predictions)
        self.assertEqual(len(coef_norms), expected_predictions)
        
        # Check that outputs are finite
        self.assertTrue(np.all(np.isfinite(predictions)))
        self.assertTrue(np.all(np.isfinite(actual_oos)))
        self.assertTrue(np.all(np.isfinite(coef_norms)))
        self.assertTrue(np.all(coef_norms >= 0))  # Norms should be non-negative
    
    def test_rolling_window_prediction_insufficient_data(self):
        """Test rolling window prediction with insufficient data."""
        short_returns = np.array([0.01, 0.02])
        short_signals = np.array([[1, 2], [3, 4]])
        window_size = 5  # Larger than data
        
        predictions, actual_oos, coef_norms = self.model.rolling_window_prediction(
            short_returns, short_signals, window_size, 10, 1.0
        )
        
        # Should return empty arrays
        self.assertEqual(len(predictions), 0)
        self.assertEqual(len(actual_oos), 0)
        self.assertEqual(len(coef_norms), 0)
    
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation."""
        # Create test data
        actual_returns = np.array([0.01, 0.02, -0.01, 0.03])
        predictions = np.array([0.015, 0.018, -0.005, 0.025])
        
        metrics = self.model.calculate_performance_metrics(predictions, actual_returns)
        
        # Check that all expected metrics are present
        expected_keys = ['r_squared', 'mean_return', 'volatility', 
                        'sharpe_ratio', 'mse', 'n_predictions']
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Check that metrics are finite
        for key in expected_keys:
            self.assertTrue(np.isfinite(metrics[key]))
        
        # Check n_predictions
        self.assertEqual(metrics['n_predictions'], len(predictions))
        
        # Test perfect prediction case
        perfect_predictions = actual_returns.copy()
        perfect_metrics = self.model.calculate_performance_metrics(
            perfect_predictions, actual_returns
        )
        self.assertAlmostEqual(perfect_metrics['r_squared'], 1.0, places=10)
        self.assertAlmostEqual(perfect_metrics['mse'], 0.0, places=10)
    
    def test_calculate_performance_metrics_zero_volatility(self):
        """Test performance metrics with zero volatility."""
        actual_returns = np.array([0.01, 0.01, 0.01])
        predictions = np.array([0.0, 0.0, 0.0])  # Zero predictions -> zero timing returns
        
        metrics = self.model.calculate_performance_metrics(predictions, actual_returns)
        
        self.assertEqual(metrics['volatility'], 0.0)
        self.assertEqual(metrics['sharpe_ratio'], 0.0)  # Should handle division by zero
    
    def test_run_analysis(self):
        """Test complete analysis run."""
        returns = self.sample_returns
        signals = self.sample_signals
        window_size = 20
        n_features = 50
        ridge_penalty = 1.0
        
        results = self.model.run_analysis(
            returns, signals, window_size, n_features, ridge_penalty
        )
        
        # Check result structure
        expected_keys = ['metrics', 'predictions', 'actual_returns', 
                        'coefficient_norms', 'parameters']
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check metrics structure
        metrics_keys = ['r_squared', 'mean_return', 'volatility', 
                       'sharpe_ratio', 'mse', 'n_predictions', 
                       'mean_coef_norm', 'complexity_ratio']
        for key in metrics_keys:
            self.assertIn(key, results['metrics'])
        
        # Check parameters
        params = results['parameters']
        self.assertEqual(params['window_size'], window_size)
        self.assertEqual(params['n_features'], n_features)
        self.assertEqual(params['ridge_penalty'], ridge_penalty)
        
        # Check complexity ratio
        expected_complexity = n_features / window_size
        self.assertAlmostEqual(results['metrics']['complexity_ratio'], 
                              expected_complexity, places=6)


class TestMainAnalysis(unittest.TestCase):
    """Test the main analysis function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        np.random.seed(42)
        n_obs = 50
        n_signals = 2
        
        sample_returns = np.random.normal(0.01, 0.1, n_obs)
        sample_signals = np.random.normal(0, 1, (n_obs, n_signals))
        
        # Create temporary CSV file
        df = pd.DataFrame({
            'returns': sample_returns,
            'signal1': sample_signals[:, 0],
            'signal2': sample_signals[:, 1]
        })
        
        self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(self.temp_csv.name, index=False)
        self.temp_csv.close()
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_csv.name):
            os.unlink(self.temp_csv.name)
    
    def test_main_analysis(self):
        """Test main analysis function."""
        # Redirect stdout to capture print output
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            results = main_analysis(
                csv_path=self.temp_csv.name,
                return_col='returns',
                signal_cols=['signal1', 'signal2'],
                window_size=12,
                n_features=100,
                ridge_penalty=1.0,
                random_seed=42
            )
            
            # Check that results are returned
            self.assertIsInstance(results, dict)
            self.assertIn('metrics', results)
            self.assertIn('parameters', results)
            
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
            
        # Check that output was printed
        output = captured_output.getvalue()
        self.assertIn('PERFORMANCE METRICS', output)
        self.assertIn('Sharpe ratio', output)
    
    def test_main_analysis_invalid_file(self):
        """Test main analysis with invalid file path."""
        with self.assertRaises(FileNotFoundError):
            main_analysis(
                csv_path='nonexistent_file.csv',
                return_col='returns', 
                signal_cols=['signal1'],
                window_size=12,
                n_features=100,
                ridge_penalty=1.0
            )


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_small_dataset(self):
        """Test with very small dataset."""
        model = VirtueOfComplexityModel(random_seed=42)
        
        # Create minimal dataset
        returns = np.array([0.01, 0.02, -0.01])
        signals = np.array([[1.0], [2.0], [3.0]])
        
        results = model.run_analysis(returns, signals, 2, 5, 1.0)
        
        # Should still work but with limited predictions
        self.assertEqual(results['metrics']['n_predictions'], 1)
    
    def test_high_complexity_ratio(self):
        """Test with very high complexity ratio (L >> T)."""
        model = VirtueOfComplexityModel(random_seed=42)
        
        returns = np.random.normal(0, 0.1, 30)
        signals = np.random.normal(0, 1, (30, 2))
        
        # High complexity: L=1000, T=5
        results = model.run_analysis(returns, signals, 5, 1000, 1.0)
        
        # Check that complexity ratio is calculated correctly
        self.assertAlmostEqual(results['metrics']['complexity_ratio'], 200.0, places=1)
        
        # Should still produce finite results
        self.assertTrue(np.isfinite(results['metrics']['sharpe_ratio']))
    
    def test_zero_ridge_penalty(self):
        """Test with zero ridge penalty (ridgeless limit)."""
        model = VirtueOfComplexityModel(random_seed=42)
        
        returns = np.random.normal(0, 0.1, 50)
        signals = np.random.normal(0, 1, (50, 3))
        
        # Test with very small ridge penalty (approaching ridgeless)
        results = model.run_analysis(returns, signals, 20, 50, 1e-6)
        
        # Should still produce finite results
        self.assertTrue(np.isfinite(results['metrics']['sharpe_ratio']))
        self.assertTrue(np.isfinite(results['metrics']['r_squared']))
    
    def test_single_signal(self):
        """Test with only one signal."""
        model = VirtueOfComplexityModel(random_seed=42)
        
        returns = np.random.normal(0, 0.1, 40)
        signals = np.random.normal(0, 1, (40, 1))
        
        results = model.run_analysis(returns, signals, 15, 30, 1.0)
        
        # Should work with single signal
        self.assertGreater(results['metrics']['n_predictions'], 0)
        self.assertTrue(np.isfinite(results['metrics']['sharpe_ratio']))
    
    def test_constant_signals(self):
        """Test with constant signals."""
        model = VirtueOfComplexityModel(random_seed=42)
        
        returns = np.random.normal(0, 0.1, 40)
        signals = np.ones((40, 2))  # Constant signals
        
        # This should work but might produce warning about constant features
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = model.run_analysis(returns, signals, 15, 30, 1.0)
        
        self.assertGreater(results['metrics']['n_predictions'], 0)


def create_sample_data_file(filename: str = "sample_data.csv", n_obs: int = 200):
    """
    Utility function to create a sample data file for testing the implementation.
    
    Args:
        filename: Name of the CSV file to create
        n_obs: Number of observations to generate
    """
    np.random.seed(42)
    
    # Generate correlated signals and returns
    # Create some predictive relationship
    signal1 = np.random.normal(0, 1, n_obs)
    signal2 = np.random.normal(0, 1, n_obs)
    signal3 = 0.3 * signal1 + 0.7 * np.random.normal(0, 1, n_obs)
    
    # Create returns with some predictable component
    noise = np.random.normal(0, 0.15, n_obs)
    returns = 0.01 + 0.05 * signal1 + 0.03 * signal2 - 0.02 * signal3 + noise
    
    # Add some momentum
    for i in range(1, n_obs):
        returns[i] += 0.1 * returns[i-1] + np.random.normal(0, 0.05)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n_obs, freq='ME'),
        'returns': returns,
        'signal1': signal1,
        'signal2': signal2, 
        'signal3': signal3
    })
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Created sample data file: {filename}")
    print(f"Contains {n_obs} observations with 3 signals")
    print(f"Columns: {list(df.columns)}")


class TestIntegration(unittest.TestCase):
    """Integration tests using more realistic scenarios."""
    
    def setUp(self):
        """Create realistic test data."""
        self.filename = "test_integration_data.csv"
        create_sample_data_file(self.filename, n_obs=120)  # 10 years of monthly data
        
    def tearDown(self):
        """Clean up test data."""
        if os.path.exists(self.filename):
            os.unlink(self.filename)
    
    def test_realistic_scenario(self):
        """Test with realistic financial data scenario."""
        # Test various parameter combinations
        test_cases = [
            {'window_size': 12, 'n_features': 100, 'ridge_penalty': 0.1},
            {'window_size': 24, 'n_features': 500, 'ridge_penalty': 1.0},
            {'window_size': 36, 'n_features': 1000, 'ridge_penalty': 10.0}
        ]
        
        for params in test_cases:
            with self.subTest(params=params):
                results = main_analysis(
                    csv_path=self.filename,
                    return_col='returns',
                    signal_cols=['signal1', 'signal2', 'signal3'],
                    **params,
                    random_seed=42
                )
                
                # Basic sanity checks
                self.assertIsInstance(results, dict)
                self.assertIn('metrics', results)
                
                metrics = results['metrics']
                
                # Check that we have reasonable number of predictions
                self.assertGreater(metrics['n_predictions'], 0)
                
                # Check that metrics are finite
                self.assertTrue(np.isfinite(metrics['r_squared']))
                self.assertTrue(np.isfinite(metrics['sharpe_ratio']))
                self.assertTrue(np.isfinite(metrics['mean_return']))
                self.assertTrue(np.isfinite(metrics['volatility']))
                
                # Volatility should be positive
                self.assertGreater(metrics['volatility'], 0)
    
    def test_complexity_comparison(self):
        """Test that higher complexity generally performs better."""
        results_low = main_analysis(
            csv_path=self.filename,
            return_col='returns',
            signal_cols=['signal1', 'signal2', 'signal3'],
            window_size=24,
            n_features=50,  # Low complexity
            ridge_penalty=1.0,
            random_seed=42
        )
        
        results_high = main_analysis(
            csv_path=self.filename,
            return_col='returns',
            signal_cols=['signal1', 'signal2', 'signal3'],
            window_size=24,
            n_features=500,  # High complexity
            ridge_penalty=1.0,
            random_seed=42
        )
        
        # Check complexity ratios
        self.assertLess(results_low['metrics']['complexity_ratio'],
                       results_high['metrics']['complexity_ratio'])
        
        # Both should produce valid results
        for results in [results_low, results_high]:
            self.assertTrue(np.isfinite(results['metrics']['sharpe_ratio']))
            self.assertGreater(results['metrics']['n_predictions'], 0)


def run_performance_benchmark():
    """
    Run a performance benchmark to test the implementation with larger datasets.
    """
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    # Create larger dataset
    benchmark_file = "benchmark_data.csv"
    create_sample_data_file(benchmark_file, n_obs=500)
    
    import time
    
    try:
        start_time = time.time()
        
        results = main_analysis(
            csv_path=benchmark_file,
            return_col='returns',
            signal_cols=['signal1', 'signal2', 'signal3'],
            window_size=60,
            n_features=2000,
            ridge_penalty=1.0,
            random_seed=42
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\nBenchmark completed in {elapsed_time:.2f} seconds")
        print(f"Processed {results['metrics']['n_predictions']} out-of-sample predictions")
        print(f"Average time per prediction: {elapsed_time/results['metrics']['n_predictions']:.4f} seconds")
        
    finally:
        if os.path.exists(benchmark_file):
            os.unlink(benchmark_file)


if __name__ == '__main__':
    # Create sample data file for manual testing
    print("Creating sample data file...")
    create_sample_data_file("sample_data.csv", n_obs=150)
    
    # Run unit tests
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)
    
    # Run tests with high verbosity
    unittest.main(argv=[''], verbosity=2, exit=False)
    
    # Run performance benchmark
    run_performance_benchmark()
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    print("\nTo run the main analysis with your own data:")
    print("1. Prepare a CSV file with returns and signal columns")
    print("2. Use the main_analysis() function or run the virtue_of_complexity.py script")
    print("3. Adjust parameters T (window_size), L (n_features), and z (ridge_penalty)")
    print("\nExample:")
    print("results = main_analysis('your_data.csv', 'return_col', ['signal1', 'signal2'])")
