import numpy as np
import timeit
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

def create_gbm_minute(init_price, mu, sigma, length):
    """
    Calculates an asset price path using the analytical solution
    to the Geometric Brownian Motion stochastic differential
    equation (SDE).

    Parameters
    ----------
    length : The int number needed to calculate length of the time series.
    init_price: Asset inital price.
    mu: The mean 'drift' of the asset.
    sigma: Volatility expressed annual terms.

    Returns
    -------
    `np.ndarray`
        The asset price path
    """
    n = length
    dt = 1 / (255 * 6.5 * 60)

    asset_path = np.exp(
        (mu - sigma ** 2 / 2) * dt +
        sigma * np.random.normal(0, np.sqrt(dt), size=n)
    )

    return init_price * asset_path.cumprod()


def Function_runtime(func, *args, **kwargs):
    start = timeit.default_timer()
    result = func(*args, **kwargs)

    stop = timeit.default_timer()
    time = stop - start
    return result, time

minutes_length = int(255*6.5*60)
num_assets = 50000
assets = np.zeros((num_assets,minutes_length))
for i in range(num_assets):
    if(i)%100 == 0:
        print(f"Generating asset path completed {int(i/num_assets*100)}%" )
    assets[i] = create_gbm_minute(100, 0.01, 0.2, minutes_length)
print(f"Generating asset path completed 100%" )

asset_easy = assets[:500,:]
asset_medium = assets[:5000,:]
asset_hard = assets

easy_runtime = Function_runtime(np.cov, asset_easy)[1]
print(f"100 assets running time: {easy_runtime}")

medium_runtime = Function_runtime(np.cov, asset_medium)[1]
print(f"1000 assets running time: {medium_runtime}")

hard_runtime = Function_runtime(np.cov, asset_hard)[1]
print(f"10000 assets running time: {hard_runtime}")