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

#reverse convertible = short down and in put + guaranted coupon 
# payoff reverse convertible= -(K-S_T)+*indatrice(min(St)<=H) + C 

# Function to generate paths using the Euler-Maruyama method
def Euler_X(X0=100, sigma=0.2, r=0, T=1, m=100, n=10**4, W=None):
    if W is None:
        W = np.random.randn(n, m)
    return X0 * np.cumprod(1 + r * T/m + sigma * np.sqrt(T/m) * W, axis=1)

# Function to estimated the price of the option by Monte Carlo simulations
def MC_RCbarrier(X0=100, sigma=0.2, r=0, T=1, K=100, H=50, C=0.1, P=100, m=100, n=10**4, W=None):
    if W is None:
        W = np.random.randn(n, m)
    X = Euler_X(X0, sigma, r, T, m, n, W) # Simulations of the n Black-Scholes paths of the underlying
    X = np.hstack((X0 * np.ones((n, 1)), X))
    min_X = np.min(X, axis=1)
    payoff_arr = -(np.maximum(K - X[:, -1], 0) * (min_X <= H)) + (1+C)*P # Payoff of the RC_barrier for each simulation
    estimator = np.mean(payoff_arr * np.exp(-r * T)) # Price estimator = discounted expectation of the payoff
    std_error = np.std(payoff_arr) / np.sqrt(n) # Standard error of the MC estimator
    return estimator, std_error


estim_1, std_error_1 = MC_RCbarrier()
print("Estimator for m = 100 : ", estim_1)
print("Confidence interval for m = 100 : [", estim_1 - 1.96*std_error_1, " ; ", estim_1 + 1.96*std_error_1, "].")


# Values of X0 to consider
X0_values = np.linspace(0, 130, 300)

# Calculate the estimated price and standard error for each X0
estimates = []
std_errors = []
for element in X0_values:
    estim, std_err = MC_RCbarrier(X0=element)
    estimates.append(estim)
    std_errors.append(std_err)

# Calculate the upper and lower bounds of the 95% confidence interval
upper_bounds = [est + 1.96 * err for est, err in zip(estimates, std_errors)]
lower_bounds = [est - 1.96 * err for est, err in zip(estimates, std_errors)]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(X0_values, estimates, label='Estimated Price')
plt.fill_between(X0_values, lower_bounds, upper_bounds, color='gray', alpha=0.2, label='95% Confidence Interval')
plt.title('Estimated Price vs. Initial Price X0 with 95% Confidence Interval')
plt.xlabel('Initial Price X0')
plt.ylabel('Estimated Price')
plt.legend()
plt.grid(True)
plt.show()

# Values of sigma to consider
sigma_values = np.linspace(0.1, 0.3, 40)

# Calculate the estimated price for each sigma
estimates = [MC_RCbarrier(sigma=element)[0] for element in sigma_values]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(sigma_values, estimates, label='Estimated Price')
plt.title('Estimated Price vs. Volatility sigma')
plt.xlabel('Volatility sigma')
plt.ylabel('Estimated Price')
plt.legend()
plt.grid(True)
plt.show()




















