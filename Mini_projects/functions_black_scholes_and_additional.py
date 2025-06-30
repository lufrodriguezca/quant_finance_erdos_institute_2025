#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.stats import norm



##Below are functions that can be used to find Black-Scholes Greeks of European Call and Put Options.


def bs_call(S0, K, sigma, t, r = 0):
    """
    Computes the Black-Scholes price of a European call option.

    Parameters:
        S0 (float): Current asset price
        K (float): Strike price
        sigma (float): Annualized volatility (standard deviation of log returns)
        t (float): Time to expiration (in years)
        r (float): Risk-free interest rate (annualized)

    Returns:
        float: Call option price
    """
    d1 = (np.log(S0/K) + (r+.5*sigma**2)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    
    call_price = S0*norm.cdf(d1) - K*np.exp(-r*t)*norm.cdf(d2)
    return call_price


def bs_put(S0, K, sigma, t, r=0):
    """
    Description:
    
    Computes the Black-Scholes value of a European put option.
    
    Parameters:
        S0: Current asset price
        K: Strike price
        sigma: Yearly standard deviation of log-returns (volatility)
        t: Time to expiration (in years)
        r: Risk-free interest rate
    
    Returns:
        Put option price
    """
    d1 = (np.log(S0/K) + (r+.5*sigma**2)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    
    put_price = -S0*norm.cdf(-d1) + K*np.exp(-r*t)*norm.cdf(-d2)
    return put_price
    
    
       


def bs_call_delta(S0, K, sigma, t, r):
    """
    Returns the Delta (sensitivity to spot price) of a European call option
    under Black-Scholes assumptions.

    Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        sigma (float): Volatility of the stock
        t (float): Time to maturity (in years)
        r (float): Risk-free interest rate

    Returns:
        float: Delta of Call Option
    """
    d1 = (np.log(S0/K) + (r+.5*sigma**2)*t)/(sigma*np.sqrt(t))
    return norm.cdf(d1)


def bs_put_delta(S0, K, sigma, t, r):
    """
    Returns the Delta (sensitivity to spot price) of a European put option
    under Black-Scholes assumptions.

    Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        sigma (float): Volatility of the stock
        t (float): Time to maturity (in years)
        r (float): Risk-free interest rate

    Returns:
        float: Delta of Put Option
    """
    d1 = (np.log(S0/K) + (r+.5*sigma**2)*t)/(sigma*np.sqrt(t))
    return norm.cdf(d1) - 1.



def bs_gamma(S0,K,sigma,t,r):
    """
    Returns the Gamma (second order rate of change with respect to spot price) of a European call and put options
    under Black-Scholes assumptions.

    Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        sigma (float): Volatility of the stock
        t (float): Time to maturity (in years)
        r (float): Risk-free interest rate

    Returns:
        float: Gamma of Call and Put Option
    """
    
    d1 = (np.log(S0/K) + (r+.5*sigma**2)*t)/(sigma*np.sqrt(t))
    
    return norm.pdf(d1)/(S0*sigma*np.sqrt(t))


def bs_vega(S0,K,sigma,t,r):
    """
    Returns the vega (rate of change with respect to volatility) of a European call and put options
    under Black-Scholes assumptions.
    
    To express vega as a percentage change, divide return by 100.

    Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        sigma (float): Volatility of the stock
        t (float): Time to maturity (in years)
        r (float): Risk-free interest rate

    Returns:
        float: Vega of Call and Put Option
    """
    
    d1 = (np.log(S0/K) + (r+.5*sigma**2)*t)/(sigma*np.sqrt(t))
    
    return S0*norm.pdf(d1)*np.sqrt(t)


def bs_call_theta(S0,K,sigma,t,r):
    """
    Returns the theta (rate of change with respect to time) of a European call option
    under Black-Scholes assumptions.
    
    
    To give daily theta, divide return by 252.

    Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        sigma (float): Volatility of the stock
        t (float): Time to maturity (in years)
        r (float): Risk-free interest rate

    Returns:
        float: Theta of Call option
    """
    
    d1 = (np.log(S0/K) + (r+.5*sigma**2)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    
    return (-S0*norm.pdf(d1)*sigma)/(2*np.sqrt(t)) - r*K*np.exp(-r*t)*norm.cdf(d2)


def bs_put_theta(S0,K,sigma,t,r):
    """
    Returns the theta (rate of change with respect to time) of a European put option
    under Black-Scholes assumptions.
    
    
    To give daily theta, divide return by 252.

    Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        sigma (float): Volatility of the stock
        t (float): Time to maturity (in years)
        r (float): Risk-free interest rate

    Returns:
        float: Theta of Put option
    """
    
    d1 = (np.log(S0/K) + (r+.5*sigma**2)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    
    return (-S0*norm.pdf(d1)*sigma)/(2*np.sqrt(t)) + r*K*np.exp(-r*t)*norm.cdf(-d2)


def bs_call_rho(S0,K,sigma,t,r):
    """
    Returns the rho (rate of change with respect to risk-free interest) of a European call option
    under Black-Scholes assumptions.
    

    Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        sigma (float): Volatility of the stock
        t (float): Time to maturity (in years)
        r (float): Risk-free interest rate

    Returns:
        float: Rho of call option
    """
    
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    
    return K*t*np.exp(-r*t)*norm.cdf(d2)


def bs_put_rho(S0,K,sigma,t,r):
    """
    Returns the rho (rate of change with respect to risk-free interest) of a European put option
    under Black-Scholes assumptions.
    

    Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        sigma (float): Volatility of the stock
        t (float): Time to maturity (in years)
        r (float): Risk-free interest rate

    Returns:
        float: Rho of put option
    """
    
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    
    return -K*t*np.exp(-r*t)*norm.cdf(-d2)


def MC_call_delta_custom(S0, K, sigma,sigma_probs, t, r, delta_sims = int(250)):
    """Description: 
    Monte-Carlo Simulation of Black-Scholes Call Delta
    
    Parameters:
    S0 (float): spot price
    K (float): strike price
    sigma (array): array of volatilities
    sigma_probs (array): probabilities of sigmas
    t (float): time to expiration
    r (float): risk-free interest rate
    delta_sims (int): Number of simulations
    
    Return
    float: simulated delta of call option
    
    """
    bump = .01*S0

    noise = np.random.normal(0,1,delta_sims)
    
    sampled_sigma = np.random.choice(sigma,p=sigma_probs,size=delta_sims)

    log_returns = (r - .5*sampled_sigma**2)*t + sampled_sigma*np.sqrt(t)*noise

    paths_up = (S0+bump)*np.exp(log_returns)
    paths_down = (S0-bump)*np.exp(log_returns)

    call_up = np.maximum(paths_up - K, 0)*np.exp(-r*t)
    call_down = np.maximum(paths_down - K, 0)*np.exp(-r*t)

    simulated_deltas = (call_up-call_down)/(2*bump)

    return np.mean(simulated_deltas)


def MC_call_delta_custom_array(S, K, sigma, sigma_probs, t, r, delta_sims=250):
    """
    Monte Carlo estimation of Black-Scholes call deltas for an array of spot prices
    with stochastic volatility model (discrete distribution).
    
    Parameters:
    S (np.array): array of spot prices
    K (float): strike price
    sigma (array): array of volatilities
    sigma_probs (array): probabilities of sigmas
    t (float): time to expiration
    r (float): risk-free interest rate
    delta_sims (int): Number of simulations
    
    Return
    float: simulated delta of call option
    """
    bump = 0.01 * S
    noise = np.random.normal(0, 1, (delta_sims, len(S)))

    sampled_sigma = np.random.choice(sigma, p=sigma_probs, size=(delta_sims, len(S)))
    log_returns = (r - 0.5 * sampled_sigma**2) * t + sampled_sigma * np.sqrt(t) * noise

    paths_up = (S + bump) * np.exp(log_returns)
    paths_down = (S - bump) * np.exp(log_returns)

    call_up = np.maximum(paths_up - K, 0) * np.exp(-r * t)
    call_down = np.maximum(paths_down - K, 0) * np.exp(-r * t)

    deltas = (call_up - call_down) / (2 * bump)
    return np.mean(deltas, axis=0)



