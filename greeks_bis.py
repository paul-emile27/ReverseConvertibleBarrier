import numpy as np
import MC_autocall

# Parameters
S0 = 100  # Initial price of the underlying asset
sigma = 0.2  # Annual volatility of the asset
r = 0.05  # Annual risk-free rate
T = 1  # Maturity of the product in years
m = 12  # Number of observation dates
n = 10**6 # Number of Monte Carlo simulations
yield_rate = 0.06  # Coupon yield rate
barrier = 0.8  # Barrier level as a percentage of the initial level


def delta(x, barrier, yield_rate, n, m, r, sigma, T, steps, epsilon=0.01):
    return (MC_autocall(x+epsilon, barrier, yield_rate, n, m, r, sigma, T) - MC_autocall(x-epsilon, barrier, yield_rate, n, m, r, sigma, T))/(2*espilon)

def gamma(x, barrier, yield_rate, n, m, r, sigma, T, steps, epsilon=0.01):
    return (MC_autocall(x+epsilon, barrier, yield_rate, n, m, r, sigma, T) - 2*MC_autocall(x, barrier, yield_rate, n, m, r, sigma, T) + MC_autocall(x-epsilon, barrier, yield_rate, n, m, r, sigma, T))/(espilon**2)

def rho(r, S0, barrier, yield_rate, n, m, sigma, T, steps, epsilon=0.002):
    return (MC_autocall(S0, barrier, yield_rate, n, m, r+epsilon, sigma, T) - MC_autocall(S0, barrier, yield_rate, n, m, r-epsilon, sigma, T))/(2*espilon)

def vega(sigma, S0, barrier, yield_rate, n, m, r, T, steps, epsilon=0.005):
        return (MC_autocall(S0, barrier, yield_rate, n, m, r, sigma+epsilon, T) - MC_autocall(S0, barrier, yield_rate, n, m, r, sigma-epsilon, T))/(2*espilon)
    
def theta(T, S0, barrier, yield_rate, n, m, r, sigma, steps, epsilon=0.01):
    return (MC_autocall(S0, barrier, yield_rate, n, m, r, sigma, T+epsilon) - MC_autocall(S0, barrier, yield_rate, n, m, r, sigma, T-epsilon))/(2*espilon)
