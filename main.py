import numpy as np
# r : free risk rate
#sigma : vol of the asset
#T maturity 
#n number of simulation of monte carlo 
#S0 initial level (intial invest)
#m : number of obs dates
#yield rate is the rate of the coupon 
#barrier : % of the initial level (usually 0.8)




# Parameters
S0 = 100  # Initial price of the underlying
sigma= 0.2  # Annual volatility
r = 0.05  # Annual drift
T = 1  # Maturity in years
m = 12  # Monthly observations
n = 1000000  # Number of Monte Carlo simulations
yield_rate = 0.06  # 6%
barrier = 0.8  # 80% of the initial level
 
def Observation(m): #generate the m observation of the asset 
    return np.concatenate((np.array([S0]), S0*np.cumprod(1 + r*T/m + sigma*np.sqrt(T/m)*np.random.randn(m))))


def autocall_payoff(S0, barrier, yield_rate, steps):
    observation = np.array([Observation(m) for i in range(n)]) #generate a matrix with n simulation of the asset at each obs date(m)
    payoffs = []
    for path in observation:
        early_termination = False
        for t in range(1, steps + 1): #start at one for first observation 
            if path[t] > S0 : #if we are above the intial level 
                payoffs.append(S0*yield_rate *t) #give the coupon I*(num of obs)
                early_termination = True
                break
        if not early_termination: #we did not reach the initial level 
            if path[-1] < barrier*S0: #case under barrier
                payoffs.append(path[-1]-S0) 
            else:
                payoffs.append(0)
    return np.array(payoffs)
 

# Simulation

payoffs = autocall_payoff(S0,barrier,yield_rate,m)*np.exp(-r*T)
 
# Calculate the expected payoff and discount it to present value
expected_payoff = np.mean(payoffs)

print(expected_payoff)

