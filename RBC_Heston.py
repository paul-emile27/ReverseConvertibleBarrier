import numpy as np 
import matplotlib.pyplot as plt
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

# Parameters relevant only for the Heston model
V0 = 0.04      # Initial variance
kappa = 1.5    # Speed of mean reversion for variance
theta = 0.06   # Long-term variance
sigmav = 0.3   # Volatility of volatility
rho = -0.6     # Correlation between the two Brownian motions

def Heston_Simulator(X0=100, V0=0.04, kappa=1.5, theta=0.06, sigmav=0.3, rho=-0.6, r=0, T=1, m=100, n=10**4, Z=None):
    dt = T / m
    X = np.zeros((n, m+1))
    V = np.zeros((n, m+1))
    X[:, 0] = X0
    V[:, 0] = V0

    # Correlated Brownian motions
    if Z is None:
        Z = np.random.normal(size=(2, n, m))
    L = np.array([[1, 0], [rho, np.sqrt(1 - rho**2)]])
    W = np.tensordot(L, Z, axes=([1],[0]))
    dWs = np.sqrt(dt) * W[0,:,:]
    dWv = np.sqrt(dt) * W[1,:,:]

    for t in range(m):
        # Ensure variance stays positive
        Vt = np.maximum(V[:, t], 0)
        
        # Euler-Maruyama for the variance process
        V[:, t+1] = V[:, t] + kappa * (theta - Vt) * dt + sigmav * np.sqrt(Vt) * dWv[:, t]
        
        # Euler-Maruyama for the stock process
        X[:, t+1] = X[:, t] + r * X[:, t] * dt + np.sqrt(Vt) * X[:, t] * dWs[:, t]

    return X

# Simulate the trajectories
heston_paths = Heston_Simulator()
plt.figure(figsize=(10, 6))
for i in range(10):  # Plot the first 10 simulated paths
    plt.plot(heston_paths[i])
plt.title('Sample Heston Model Paths')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.show()

def MC_RCbarrier_Heston(X0=100, V0=0.04, kappa=1.5, theta=0.06, sigmav=0.3, rho=-0.6, r=0, T=1, K=100, H=50, C=0.1, P=100, m=100, n=10**4, Z=None):
    if Z is None:
        Z = np.random.normal(size=(2, n, m))
    # Generate the paths using the Heston model
    X = Heston_Simulator(X0, V0, kappa, theta, sigmav, rho, r, T, m, n, Z)
    
    # Calculate the minimum value of each path
    min_X = np.min(X, axis=1)

    # Calculate the payoff for each simulation
    payoff_arr = -(np.maximum(K - X[:, -1], 0) * (min_X <= H)) + (1 + C) * P
    
    # Calculate the estimator and standard error
    estimator = np.mean(payoff_arr * np.exp(-r * T))
    std_error = np.std(payoff_arr) / np.sqrt(n)
    
    return estimator, std_error

estim_2, std_error_2 = MC_RCbarrier_Heston()
print("Estimator for m = 100 : ", estim_2)
print("Confidence interval for m = 100 : [", estim_2 - 1.96*std_error_2, " ; ", estim_2 + 1.96*std_error_2, "].")


# Calculate the estimated price and standard error for each X0
estimates_heston = []
std_errors_heston = []
for element in X0_values:
    estim, std_err = MC_RCbarrier_Heston(X0=element)
    estimates_heston.append(estim)
    std_errors_heston.append(std_err)

# Calculate the upper and lower bounds of the 95% confidence interval
upper_bounds_heston = [est + 1.96 * err for est, err in zip(estimates_heston, std_errors_heston)]
lower_bounds_heston = [est - 1.96 * err for est, err in zip(estimates_heston, std_errors_heston)]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(X0_values, estimates_heston, label='Estimated Price')
plt.fill_between(X0_values, lower_bounds_heston, upper_bounds_heston, color='gray', alpha=0.2, label='95% Confidence Interval')
plt.title('Estimated Price vs. Initial Price X0 with 95% Confidence Interval for the Heston model')
plt.xlabel('Initial Price X0')
plt.ylabel('Estimated Price')
plt.legend()
plt.grid(True)
plt.show()
















