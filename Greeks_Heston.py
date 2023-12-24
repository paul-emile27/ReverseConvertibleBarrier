import numpy as np 
import matplotlib.pyplot as plt
import MC_RCbarrier_Heston 
np.random.seed(40)
i


def calculate_delta_Heston(X0=100, V0=0.04, kappa=1.5, theta=0.06, sigmav=0.3, rho=-0.6, r=0, T=1, K=100, H=50, C=0.1, P=100, m=100, n=10**4, delta_X=1, Z=None):
    if Z is None:
        Z = np.random.normal(size=(2, n, m))
    price_up = MC_RCbarrier_Heston(X0+delta_X, V0, kappa, theta, sigmav, rho, r, T, K, H, C, P, m, n, Z)[0]
    price_down = MC_RCbarrier_Heston(X0-delta_X, V0, kappa, theta, sigmav, rho, r, T, K, H, C, P, m, n, Z)[0]
    return (price_up - price_down) / (2 * delta_X)

def calculate_gamma_Heston(X0=100, V0=0.04, kappa=1.5, theta=0.06, sigmav=0.3, rho=-0.6, r=0, T=1, K=100, H=50, C=0.1, P=100, m=100, n=10**4, delta_X=1, Z=None):
    if Z is None:
        Z = np.random.normal(size=(2, n, m))
    price_up = MC_RCbarrier_Heston(X0+delta_X, V0, kappa, theta, sigmav, rho, r, T, K, H, C, P, m, n, Z)[0]
    price_down = MC_RCbarrier_Heston(X0-delta_X, V0, kappa, theta, sigmav, rho, r, T, K, H, C, P, m, n, Z)[0]
    base_price = MC_RCbarrier_Heston(X0, V0, kappa, theta, sigmav, rho, r, T, K, H, C, P, m, n, Z)[0]
    return (price_up - 2 * base_price + price_down) / (delta_X ** 2)

def calculate_vega_Heston(X0=100, V0=0.04, kappa=1.5, theta=0.06, sigmav=0.3, rho=-0.6, r=0, T=1, K=100, H=50, C=0.1, P=100, m=100, n=10**4, delta_sigmav=0.05, Z=None):
    if Z is None:
        Z = np.random.normal(size=(2, n, m))
    base_price = MC_RCbarrier_Heston(X0, V0, kappa, theta, sigmav, rho, r, T, K, H, C, P, m, n, Z)[0]
    price_vol_up = MC_RCbarrier_Heston(X0, V0, kappa, theta, sigmav+delta_sigmav, rho, r, T, K, H, C, P, m, n, Z)[0]
    return (price_vol_up - base_price) / delta_sigmav


# Calculate the estimated Delta and Gamma for each X0
deltas = [calculate_delta_Heston(X0=element) for element in X0_values]
gammas = [calculate_gamma_Heston(X0=element) for element in X0_values]

# Create a figure and a set of subplots
plt.figure(figsize=(18, 5))

# Delta Plot
plt.subplot(1, 2, 1)  # 1 row, 3 columns, 1st subplot
plt.plot(X0_values, deltas, label='Estimated Delta', color='blue')
plt.title('Estimated Delta vs. Initial Price X0')
plt.xlabel('Initial Price X0')
plt.ylabel('Estimated Delta')
plt.legend()
plt.grid(True)

# Gamma Plot
plt.subplot(1, 2, 2)  # 1 row, 3 columns, 2nd subplot
plt.plot(X0_values, gammas, label='Estimated Gamma', color='green')
plt.title('Estimated Gamma vs. Initial Price X0')
plt.xlabel('Initial Price X0')
plt.ylabel('Estimated Gamma')
plt.legend()
plt.grid(True)

# Display the plots
plt.tight_layout()
plt.show()
