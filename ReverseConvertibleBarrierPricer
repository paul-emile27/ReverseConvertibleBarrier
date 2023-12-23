import numpy as np 
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

r,sigma,X0,T,n,m = 0 , 0.2 , 100 , 1 , 10**6 , 1000 

"""
Parameters of the DI Put option and the coupon :

K : strike of the put option (usually ATM)
H : barrier (usually 50-80% of X0)
C : coupon rate (usually 6%)
P : principal invested
"""
K , H , C , P = 100 , 80 , 0.1 , 100

#Generate Euler Scheme for geometric brownian motion 

def Euler_X(r,sigma,X0,T,m):
    return np.concatenate((np.array([X0]), X0*np.cumprod(1 + r*T/m + sigma*np.sqrt(T/m)*np.random.randn(m))))



def MC_RCbarrier(r,sigma,X0,T,n,m ,K , H , C , P): 
    
    X = np.array([Euler_X(r,sigma,X0,T,m) for i in range(n)]) #Generate a matrix (n,m+1) of n simulations for Euler Scheme with m+1 steps
    
    min_X = np.min(X, axis=1) #Get the min for each simulation 
    payoff_arr = -(np.maximum(K- X[:, -1] , 0) * (min_X < H)) + (1+P)*C #Get the payoff of RC Barrier for each simulation 

    estimator = np.mean(payoff_arr*np.exp(-r*T)) #Compute MC estimator 
    
    
    std_error = np.std(payoff_arr) / np.sqrt(n) #Compute MC error 
    
    return estimator, std_error




estim_1, std_error_1 = MC_RCbarrier(r,sigma,X0,T,n,m,K , H , C , P) 

print("Estimator for m : "  , estim_1) 

#Confidence interval of level 95%
print("Confidence interval for m = 100 : [", estim_1 - 1.96*std_error_1, " ; ", estim_1 + 1.96*std_error_1, "].") 


