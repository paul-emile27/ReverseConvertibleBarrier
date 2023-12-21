import numpy as np

# Parameters
S0 = 100  # Initial price of the underlying asset
sigma = 0.2  # Annual volatility of the asset
r = 0.05  # Annual risk-free rate
T = 1  # Maturity of the product in years
m = 12  # Number of observation dates
n = 10**6 # Number of Monte Carlo simulations
yield_rate = 0.06  # Coupon yield rate
barrier = 0.8  # Barrier level as a percentage of the initial level

def Observation(n, m, S0, r, sigma, T):
    """ 
    Generate a matrix (n, m+1) of m observations  for n samples of the asset.
    This function simulates the asset price paths for the Monte Carlo simulation.
    """
    # Initialize the matrix with zeros
    Obs = np.zeros((n, m + 1))  # +1 to include the initial price
    # Fill the matrix
    for i in range(n):
        # Generate each path with random fluctuations
        path = S0 * np.cumprod(1 + r * T/m + sigma * np.sqrt(T/m) * np.random.randn(m))
        # Insert the initial price at the beginning of each path
        Obs[i, :] = np.concatenate(([S0], path))
    return Obs

def autocall_payoff(S0, barrier, yield_rate, n, m, r, sigma, T):
    """
    Calculate the payoff for each simulation path of the autocallable product.
    This function determines the payoff based on the occurrence of early termination and the final asset price relative to the barrier.
    """
    observation = Observation(n, m, S0, r, sigma, T)  # Generate asset paths
    payoffs = []
    for path in observation:
        early_termination = False
        for t in range(1, m + 1):  # Loop over each observation date
            if path[t] > S0:  # Check for early termination condition
                payoffs.append((1 + yield_rate * t))  # Calculate payoff for early termination
                early_termination = True
                break
        if not early_termination:  # If no early termination
            if path[-1] < barrier * S0:  # If asset price is below the barrier at maturity
                payoffs.append(path[-1] / S0)  # Payoff is the negative performance
            else:  # If asset price is above the barrier
                payoffs.append(1)  # No payoff as the asset price is above the barrier
    return np.array(payoffs)

def MC_autocall(S0, barrier, yield_rate, n, m, r, sigma, T):
    """
    Perform Monte Carlo simulation to calculate the average expected payoff of the autocallable product.
    This function averages the payoffs from all simulation paths and discounts them to present value.
    """
    payoffs = autocall_payoff(S0, barrier, yield_rate, n, m, r, sigma, T)  # Calculate payoffs for each path
    return np.mean(payoffs) * np.exp(-r * T)  # Discount payoffs to present value and calculate average


