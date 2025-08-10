import pandas as pd
import numpy as np

def make_test_data_csv(filename="test_returns.csv", n=240, seed=42):
    rng = np.random.default_rng(seed)
    # Monthly date index
    dates = pd.date_range("2000-01-31", periods=n, freq="M")
    # Simulate 3 predictors (signals)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    # Non-linear “true” relationship for returns
    y_true = np.sin(1.5 * x1) + 0.3 * x2**2 - 0.4 * x3
    returns = y_true + rng.normal(scale=0.5, size=n)
    # Put into DataFrame
    df = pd.DataFrame({
        "date": dates,
        "return": returns,
        "sig1": x1,
        "sig2": x2,
        "sig3": x3
    })
    df.to_csv(filename, index=False)
    print(f"Wrote {filename} with shape {df.shape}")

if __name__ == "__main__":
    make_test_data_csv()
