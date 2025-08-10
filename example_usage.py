#!/usr/bin/env python3
"""
Example usage script for the Virtue of Complexity implementation.

This script demonstrates how to:
1. Create sample financial data
2. Run the virtue of complexity analysis
3. Interpret the results
4. Experiment with different parameters

Author: Based on Kelly, Malamud, and Zhou (2022)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from virtue_of_complexity import VirtueOfComplexityModel, main_analysis

def create_realistic_financial_data(n_obs=240, filename="financial_data.csv"):
    """
    Create realistic financial time series data for testing.
    
    Args:
        n_obs: Number of observations (e.g., 240 for 20 years of monthly data)
        filename: Output CSV filename
    
    Returns:
        DataFrame with the created data
    """
    np.random.seed(42)
    
    print(f"Creating {n_obs} observations of synthetic financial data...")
    
    # Create date index
    dates = pd.date_range('2000-01-01', periods=n_obs, freq='ME')
    
    # Create correlated macro-economic signals (similar to Welch-Goyal predictors)
    dividend_yield = np.random.normal(0.03, 0.01, n_obs)  # Dividend yield
    term_spread = np.random.normal(0.02, 0.005, n_obs)    # Term structure spread
    default_spread = np.random.normal(0.01, 0.003, n_obs) # Default spread
    book_to_market = np.random.normal(0.7, 0.2, n_obs)   # Book-to-market ratio
    earnings_yield = dividend_yield + np.random.normal(0.02, 0.01, n_obs)
    
    # Add some persistence (AR(1) structure)
    for i in range(1, n_obs):
        dividend_yield[i] += 0.8 * dividend_yield[i-1] * 0.1
        term_spread[i] += 0.7 * term_spread[i-1] * 0.1
        default_spread[i] += 0.6 * default_spread[i-1] * 0.1
        book_to_market[i] += 0.5 * book_to_market[i-1] * 0.1
        earnings_yield[i] += 0.7 * earnings_yield[i-1] * 0.1
    
    # Create stock market returns with predictable component
    base_return = 0.008  # 0.8% monthly base return
    
    # Returns depend on lagged predictors (as in the literature)
    returns = np.zeros(n_obs)
    for i in range(1, n_obs):
        predictable_component = (
            0.3 * dividend_yield[i-1] +
            0.2 * term_spread[i-1] + 
            -0.4 * default_spread[i-1] +
            0.1 * book_to_market[i-1] +
            0.15 * earnings_yield[i-1]
        )
        
        # Add momentum effect
        momentum = 0.1 * returns[i-1] if i > 1 else 0
        
        # Add noise
        noise = np.random.normal(0, 0.04)  # 4% monthly volatility
        
        returns[i] = base_return + predictable_component + momentum + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'returns': returns,
        'dividend_yield': dividend_yield,
        'term_spread': term_spread,
        'default_spread': default_spread,
        'book_to_market': book_to_market,
        'earnings_yield': earnings_yield
    })
    
    # Save to CSV
    df.to_csv(filename, index=False)
    
    # Print summary statistics
    print(f"\nData Summary:")
    print(f"Date range: {dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')}")
    print(f"Mean monthly return: {np.mean(returns):.4f} ({np.mean(returns)*12:.1%} annualized)")
    print(f"Monthly volatility: {np.std(returns):.4f} ({np.std(returns)*np.sqrt(12):.1%} annualized)")
    print(f"Sharpe ratio: {np.mean(returns)/np.std(returns)*np.sqrt(12):.2f}")
    
    return df

def compare_complexity_levels(csv_path, return_col, signal_cols):
    """
    Compare performance across different complexity levels (L/T ratios).
    
    This demonstrates the "virtue of complexity" phenomenon.
    """
    print("\n" + "="*60)
    print("COMPARING DIFFERENT COMPLEXITY LEVELS")
    print("="*60)
    
    # Test different complexity levels
    complexity_tests = [
        {'T': 60, 'L': 30, 'description': 'Low complexity (L < T)'},
        {'T': 60, 'L': 60, 'description': 'Interpolation boundary (L = T)'},  
        {'T': 60, 'L': 300, 'description': 'High complexity (L >> T)'},
        {'T': 60, 'L': 1000, 'description': 'Very high complexity'}
    ]
    
    results = []
    
    for test in complexity_tests:
        print(f"\nTesting {test['description']}:")
        print(f"Window size (T): {test['T']}, Features (L): {test['L']}")
        print(f"Complexity ratio (L/T): {test['L']/test['T']:.1f}")
        
        result = main_analysis(
            csv_path=csv_path,
            return_col=return_col,
            signal_cols=signal_cols,
            window_size=test['T'],
            n_features=test['L'],
            ridge_penalty=1.0,  # Use moderate regularization
            random_seed=42
        )
        
        metrics = result['metrics']
        results.append({
            'complexity': test['L']/test['T'],
            'description': test['description'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'r_squared': metrics['r_squared'],
            'mean_return': metrics['mean_return'],
            'volatility': metrics['volatility']
        })
    
    # Summary table
    print("\n" + "="*80)
    print("COMPLEXITY COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Description':<25} {'L/T':<8} {'Sharpe':<8} {'R²':<8} {'Return':<8} {'Vol':<8}")
    print("-"*80)
    
    for r in results:
        print(f"{r['description']:<25} {r['complexity']:<8.1f} {r['sharpe_ratio']:<8.3f} "
              f"{r['r_squared']:<8.3f} {r['mean_return']:<8.4f} {r['volatility']:<8.3f}")
    
    return results

def test_regularization_effects(csv_path, return_col, signal_cols):
    """
    Test the effect of different regularization parameters.
    """
    print("\n" + "="*60)
    print("TESTING REGULARIZATION EFFECTS")
    print("="*60)
    
    # Test different ridge penalty values
    ridge_penalties = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    results = []
    
    for penalty in ridge_penalties:
        print(f"\nTesting ridge penalty: {penalty}")
        
        result = main_analysis(
            csv_path=csv_path,
            return_col=return_col,
            signal_cols=signal_cols,
            window_size=36,
            n_features=500,  # High complexity
            ridge_penalty=penalty,
            random_seed=42
        )
        
        metrics = result['metrics']
        results.append({
            'ridge_penalty': penalty,
            'sharpe_ratio': metrics['sharpe_ratio'],
            'r_squared': metrics['r_squared'],
            'mean_coef_norm': metrics['mean_coef_norm']
        })
    
    # Summary
    print("\n" + "="*50)
    print("REGULARIZATION SUMMARY")
    print("="*50)
    print(f"{'Ridge λ':<10} {'Sharpe':<8} {'R²':<8} {'||β||':<10}")
    print("-"*50)
    
    for r in results:
        print(f"{r['ridge_penalty']:<10.3f} {r['sharpe_ratio']:<8.3f} "
              f"{r['r_squared']:<8.3f} {r['mean_coef_norm']:<10.3f}")
    
    return results

def demonstrate_reproducibility(csv_path, return_col, signal_cols):
    """
    Demonstrate that results are reproducible with the same random seed.
    """
    print("\n" + "="*60)
    print("TESTING REPRODUCIBILITY")
    print("="*60)
    
    # Run same analysis twice with same seed
    result1 = main_analysis(
        csv_path=csv_path, return_col=return_col, signal_cols=signal_cols,
        window_size=24, n_features=200, ridge_penalty=1.0, random_seed=42
    )
    
    result2 = main_analysis(
        csv_path=csv_path, return_col=return_col, signal_cols=signal_cols,
        window_size=24, n_features=200, ridge_penalty=1.0, random_seed=42
    )
    
    # Check if results are identical
    metrics1 = result1['metrics']
    metrics2 = result2['metrics']
    
    print("Comparing results from two identical runs:")
    print(f"Sharpe ratio difference: {abs(metrics1['sharpe_ratio'] - metrics2['sharpe_ratio']):.10f}")
    print(f"R² difference: {abs(metrics1['r_squared'] - metrics2['r_squared']):.10f}")
    
    if abs(metrics1['sharpe_ratio'] - metrics2['sharpe_ratio']) < 1e-10:
        print("✓ Results are perfectly reproducible!")
    else:
        print("⚠ Results are not identical - check random seed implementation")

def main():
    """
    Main demonstration function.
    """
    print("="*60)
    print("VIRTUE OF COMPLEXITY - DEMONSTRATION SCRIPT")
    print("Based on Kelly, Malamud, and Zhou (2022)")
    print("="*60)
    
    # Step 1: Create sample data
    data_file = "demo_financial_data.csv"
    df = create_realistic_financial_data(n_obs=200, filename=data_file)
    
    return_col = 'returns'
    signal_cols = ['dividend_yield', 'term_spread', 'default_spread', 'book_to_market', 'earnings_yield']
    
    # Step 2: Run basic analysis
    print("\n" + "="*60)
    print("BASIC ANALYSIS")
    print("="*60)
    
    basic_result = main_analysis(
        csv_path=data_file,
        return_col=return_col,
        signal_cols=signal_cols,
        window_size=36,
        n_features=500,
        ridge_penalty=1.0,
        random_seed=42
    )
    
    # Step 3: Compare complexity levels
    complexity_results = compare_complexity_levels(data_file, return_col, signal_cols)
    
    # Step 4: Test regularization
    regularization_results = test_regularization_effects(data_file, return_col, signal_cols)
    
    # Step 5: Test reproducibility  
    demonstrate_reproducibility(data_file, return_col, signal_cols)
    
    # Step 6: Final recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("\n1. VIRTUE OF COMPLEXITY:")
    print("   - Higher complexity (L >> T) generally improves performance")
    print("   - The 'virtue' is that more parameters can be better, even with limited data")
    
    print("\n2. REGULARIZATION:")
    print("   - Use moderate ridge penalties (λ ≈ 1.0) for best results")
    print("   - Too little regularization causes instability near interpolation boundary")
    print("   - Too much regularization reduces model flexibility")
    
    print("\n3. PRACTICAL USAGE:")
    print("   - Start with T=36 (3 years), L=500-1000, λ=1.0")
    print("   - Experiment with different complexity ratios")
    print("   - Ensure you have at least T+20 observations for meaningful results")
    
    print("\n4. DATA REQUIREMENTS:")
    print("   - Standardize your signals (done automatically)")
    print("   - Handle missing values before analysis")
    print("   - Consider signal relevance and economic intuition")
    
    # Cleanup
    try:
        import os
        os.unlink(data_file)
        print(f"\nCleaned up temporary file: {data_file}")
    except:
        pass
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
