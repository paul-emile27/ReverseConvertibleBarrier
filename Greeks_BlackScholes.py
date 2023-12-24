import numpy as np 
import matplotlib.pyplot as plt
import Euler_X 
import MC_RCbarrier

np.random.seed(40)

""""
Parameters used :
r : risk free rate 
sigma : volatility of the stock 
X0 : initial stock price
T : maturity date
n : number of simulation 
m : step in Euler scheme
A reverse convertible can be seen as a short down an in put (barrier option) and a guaranteed coupon received at maturity.
"""

r,sigma,X0,T,n,m = 0 , 0.2 , 100 , 1 , 10**4, 100 

"""
Parameters of the DI Put option and the coupon :

K : strike of the put option (usually ATM)
H : barrier (usually 50-80% of X0)
C : coupon rate (usually 6%)
P : principal invested
"""
K , H , C , P = 100 , 80 , 0.1 , 100

def calculate_delta(X0=100, sigma=0.2, T=1, K=100, H=50, C=0.1, P=100, m=100, n=10**4, delta_X=1, W=None):
    if W is None:
        W = np.random.randn(n, m)
    price_up = MC_RCbarrier(X0+delta_X, sigma, T, K, H, C, P, m, n, W)[0]
    price_down = MC_RCbarrier(X0-delta_X, sigma,  T, K, H, C, P, m, n, W)[0]
    return (price_up - price_down) / (2 * delta_X)

def calculate_gamma(X0=100, sigma=0.2, T=1, K=100, H=50, C=0.1, P=100, m=100, n=10**4, delta_X=2, W=None):
    if W is None:
        W = np.random.randn(n, m)
    price_up = MC_RCbarrier(X0+delta_X, sigma, T, K, H, C, P, m, n, W)[0]
    price_down = MC_RCbarrier(X0-delta_X, sigma,  T, K, H, C, P, m, n, W)[0]
    base_price = MC_RCbarrier(X0, sigma,  T, K, H, C, P, m, n, W)[0]
    return (price_up - 2 * base_price + price_down) / (delta_X ** 2)

def calculate_vega(X0=100, sigma=0.2, T=1, K=100, H=50, C=0.1, P=100, m=100, n=10**4, delta_sigma=0.05, W=None):
    if W is None:
        W = np.random.randn(n, m)
    base_price = MC_RCbarrier(X0, sigma, T, K, H, C, P, m, n, W)[0]
    price_vol_up = MC_RCbarrier(X0, sigma+delta_sigma, T, K, H, C, P, m, n, W)[0]
    return (price_vol_up - base_price) / delta_sigma

def calculate_theta(X0=100, sigma=0.2, T=1, K=100, H=50, C=0.1, P=100, m=100, n=10**4, delta_t=0.02, W=None):
    if W is None:
        W = np.random.randn(n, m)
    base_price = MC_RCbarrier(X0, sigma, T, K, H, C, P, m, n, W)[0]
    price_t_up = MC_RCbarrier(X0, sigma, T+delta_t, K, H, C, P, m, n, W)[0]
    return (base_price - price_t_up) / delta_t

def calculate_rho(X0=100, sigma=0.2, r=0, T=1, K=100, H=50, C=0.1, P=100, m=100, n=10**4, delta_r=0.005, W=None):
    if W is None:
        W = np.random.randn(n, m)
    base_price = MC_RCbarrier(X0, sigma, r, T, K, H, C, P, m, n, W)[0]
    price_r_up = MC_RCbarrier(X0, sigma, r+delta_r, T, K, H, C, P, m, n, W)[0]
    return (price_r_up - base_price) / delta_r

Z = np.random.randn(n, m)
# Calculate the estimated Delta for each X0
deltas = [calculate_delta(X0=element, W=Z) for element in X0_values]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(X0_values, deltas, label='Estimated Delta')
plt.title('Estimated Delta vs. Initial Price X0')
plt.xlabel('Initial Price X0')
plt.ylabel('Estimated Delta')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the estimated Gamma for each X0
gammas = [calculate_gamma(X0=element, W=Z) for element in X0_values]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(X0_values, gammas, label='Estimated Gamma')
plt.title('Estimated Gamma vs. Initial Price X0')
plt.xlabel('Initial Price X0')
plt.ylabel('Estimated Gamma')
plt.legend()
plt.grid(True)
plt.show()


# Calculate the estimated vega for each sigma
vegas = [calculate_vega(X0=element, W=Z) for element in X0_values]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(X0_values, vegas, label='Estimated Vega')
plt.title('Estimated Vega vs. Initial price X0')
plt.xlabel('X0')
plt.ylabel('Estimated Vega')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the estimated thetas for each X0
thetas = [calculate_theta(X0=element, W=Z) for element in X0_values]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(X0_values, thetas, label='Estimated Theta')
plt.title('Estimated Theta vs. Initial Price X0')
plt.xlabel('Initial Price X0')
plt.ylabel('Estimated Theta')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the estimated rhos for each X0
rhos = [calculate_rho(X0=element, W=Z) for element in X0_values]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(X0_values, rhos, label='Estimated Rho')
plt.title('Estimated Rho vs. Initial Price X0')
plt.xlabel('Initial Price X0')
plt.ylabel('Estimated Rho')
plt.legend()
plt.grid(True)
plt.show()


















