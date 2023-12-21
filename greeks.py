{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "0b0f6d27",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import MC_autocall\n",
    "\n",
    "# Parameters\n",
    "S0 = 100  # Initial price of the underlying asset\n",
    "sigma = 0.2  # Annual volatility of the asset\n",
    "r = 0.05  # Annual risk-free rate\n",
    "T = 1  # Maturity of the product in years\n",
    "m = 12  # Number of observation dates\n",
    "n = 10**6 # Number of Monte Carlo simulations\n",
    "yield_rate = 0.06  # Coupon yield rate\n",
    "barrier = 0.8  # Barrier level as a percentage of the initial level\n",
    "\n",
    "\n",
    "def delta(x, barrier, yield_rate, n, m, r, sigma, T, steps, epsilon=0.01):\n",
    "    return (MC_autocall(x+epsilon, barrier, yield_rate, n, m, r, sigma, T) - MC_autocall(x-epsilon, barrier, yield_rate, n, m, r, sigma, T))/(2*espilon)\n",
    "\n",
    "def gamma(x, barrier, yield_rate, n, m, r, sigma, T, steps, epsilon=0.01):\n",
    "    return (MC_autocall(x+epsilon, barrier, yield_rate, n, m, r, sigma, T) - 2*MC_autocall(x, barrier, yield_rate, n, m, r, sigma, T) + MC_autocall(x-epsilon, barrier, yield_rate, n, m, r, sigma, T))/(espilon**2)\n",
    "\n",
    "def rho(r, S0, barrier, yield_rate, n, m, sigma, T, steps, epsilon=0.002):\n",
    "    return (MC_autocall(S0, barrier, yield_rate, n, m, r+epsilon, sigma, T) - MC_autocall(S0, barrier, yield_rate, n, m, r-epsilon, sigma, T))/(2*espilon)\n",
    "\n",
    "def vega(sigma, S0, barrier, yield_rate, n, m, r, T, steps, epsilon=0.005):\n",
    "        return (MC_autocall(S0, barrier, yield_rate, n, m, r, sigma+epsilon, T) - MC_autocall(S0, barrier, yield_rate, n, m, r, sigma-epsilon, T))/(2*espilon)\n",
    "    \n",
    "def theta(T, S0, barrier, yield_rate, n, m, r, sigma, steps, epsilon=0.01):\n",
    "    return (MC_autocall(S0, barrier, yield_rate, n, m, r, sigma, T+epsilon) - MC_autocall(S0, barrier, yield_rate, n, m, r, sigma, T-epsilon))/(2*espilon)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}