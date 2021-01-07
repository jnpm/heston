import math
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as scs
from kaleido.scopes.plotly import PlotlyScope
from numpy import matlib
from optimparallel import minimize_parallel
from plotly.subplots import make_subplots
from rich.console import Console
from rich.table import Table
from scipy.optimize import differential_evolution
from scipy.special import comb

# 32-point Gauss-Laguerre Quadrature
n = 32


## Generate abscissas (x) and weights (w) for Gauss-Laguerre integration
def GenerateGaussLaguerre(n):
    L = np.empty(n + 1)
    dL = np.empty([n, n])

    for k in range(0, n + 1):
        # Laguerre polynomial of order N
        L[k] = (((-1) ** k) / math.factorial(k)) * comb(n, k)
    # flip vector to get roots in correct order
    L_flipped = np.flip(L)

    # roots of polynomial
    x = np.flipud(np.roots(L_flipped))

    w = np.zeros((n, 1))
    # weights obtained with the derivative of the Laguerre polynomial evaluated at each of the N abscissas
    for j in range(0, n):
        for k in range(0, n):
            dL[k, j] = ((-1) ** (k + 1) / math.factorial(k - 1 + 1)) * comb(n, k + 1) * x[j] ** (k - 1 + 1)
        w[j] = (np.exp(x[j]) / x[j]) / (np.sum(dL[:, j])) ** 2
    return x, w


## Method 1: Standard way calculating 2 integrals
## Returns the integrand for the risk neutral probabilities P1 and P2
def HestonProb(phi, kappa, theta, lmbda, rho, sigma, tau, K, S, r, q, v, Pnum, Trap):
    x = np.log(S)
    a = kappa * theta

    if Pnum == 1:
        u = 0.5
        b = kappa + lmbda - (rho * sigma)
    else:
        u = -0.5
        b = kappa + lmbda

    d = np.sqrt((rho * sigma * 1j * phi - b) ** 2 - sigma ** 2 * (2 * u * 1j * phi - phi ** 2))
    g = (b - rho * sigma * 1j * phi + d) / (b - rho * sigma * 1j * phi - d)

    if Trap == 1:
        # Little Heston trap formulation
        c = 1 / g
        D = (b - rho * sigma * 1j * phi - d) / sigma ** 2 * ((1 - np.exp(-d * tau)) / (1 - c * np.exp(-d * tau)))
        G = (1 - c * np.exp(-d * tau)) / (1 - c)
        C = (r - q) * 1j * phi * tau + a / sigma ** 2 * ((b - rho * sigma * 1j * phi - d) * tau - 2 * np.log(G))
    elif Trap == 0:
        G = (1 - g * np.exp(d * tau)) / (1 - g)
        C = (r - q) * 1j * phi * tau + a / sigma ** 2 * ((b - rho * sigma * 1j * phi + d) * tau - 2 * np.log(G))
        D = (b - rho * sigma * 1j * phi + d) / sigma ** 2 * ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau)))

    # characteristic function
    f = np.exp(C + D * v + 1j * phi * x)

    # Return real part of the integrand
    y = np.real(np.exp(-1j * phi * np.log(K)) * f / 1j / phi)

    return y


## Method 2: Consolidated integrals (1 integral)
## Returns the integrand for the regrouped risk neutral probability
def HestonProbConsol(phi, kappa, theta, lmbda, rho, sigma, tau, K, S, r, q, v, Trap):
    x = np.log(S)
    a = kappa * theta
    u = -0.5
    b = kappa + lmbda

    d = np.sqrt((rho * sigma * 1j * phi - b) ** 2 - sigma ** 2 * (2 * u * 1j * phi - phi ** 2))
    g = (b - rho * sigma * 1j * phi + d) / (b - rho * sigma * 1j * phi - d)

    if Trap == 1:
        # Little Heston trap formulation
        c = 1 / g
        D = (b - rho * sigma * 1j * phi - d) / sigma ** 2 * ((1 - np.exp(-d * tau)) / (1 - c * np.exp(-d * tau)))
        G = (1 - c * np.exp(-d * tau)) / (1 - c)
        C = (r - q) * 1j * phi * tau + a / sigma ** 2 * ((b - rho * sigma * 1j * phi - d) * tau - 2 * np.log(G))
    elif Trap == 0:
        G = (1 - g * np.exp(d * tau)) / (1 - g)
        C = (r - q) * 1j * phi * tau + a / sigma ** 2 * ((b - rho * sigma * 1j * phi + d) * tau - 2 * np.log(G))
        D = (b - rho * sigma * 1j * phi + d) / sigma ** 2 * ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau)))

    # Return real part of the integrand
    y = np.exp(C + D * v + 1j * phi * x)

    return y


## Computes Heston (1993) call or put price using Gauss-Laguerre Quadrature
def HestonPriceGaussLaguerre(PutCall, S, K, T, rf, q, param, trap, x, w, IntMethod):
    kappa = param[0]
    theta = param[1]
    sigma = param[2]
    v0 = param[3]
    rho = param[4]
    lmbda = 0

    int = np.empty(n)

    int1 = np.empty(n)
    int2 = np.empty(n)

    # Numerical integration
    for k in range(0, len(x)):
        if IntMethod == 1:
            int1[k] = w[k] * HestonProb(x[k], kappa, theta, lmbda, rho, sigma, T, K, S, rf, q, v0, 1, trap)
            int2[k] = w[k] * HestonProb(x[k], kappa, theta, lmbda, rho, sigma, T, K, S, rf, q, v0, 2, trap)
        else:
            f1 = HestonProbConsol(x[k] - 1j, kappa, theta, lmbda, rho, sigma, T, K, S, rf, q, v0, trap)
            f2 = HestonProbConsol(x[k], kappa, theta, lmbda, rho, sigma, T, K, S, rf, q, v0, trap)
            int[k] = w[k] * np.real(np.exp(-1j * x[k] * np.log(K) - rf * T) / 1j / x[k] * (f1 - K * f2))

    if IntMethod == 1:
        P1 = 0.5 + ((1 / math.pi) * np.sum(int1))
        P2 = 0.5 + ((1 / math.pi) * np.sum(int2))
        # Call Price
        HestonC = S * np.exp(-q * T) * P1 - K * np.exp(-rf * T) * P2
    else:
        Integral = np.sum(int)
        # Call Price
        HestonC = 0.5 * S * np.exp(-q * T) - 0.5 * K * np.exp(-rf * T) + (1 / math.pi) * Integral

    if PutCall == 'C':
        y = HestonC
    else:
        # Put Price using C-P-P
        HestonP = HestonC - S * np.exp(-q * T) + K * np.exp(-rf * T)
        y = HestonP

    return y


## Closed form Black-Scholes formula for Call options
def EuCall_BlackScholes(S, K, rf, q, sigma, T):
    d_1 = (np.log(S / K) + (rf - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d_2 = (np.log(S / K) + (rf - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    phi = scs.norm.cdf(d_1)
    C = S * np.exp(-q * (T)) * phi - K * np.exp(-rf * (T)) * scs.norm.cdf(d_2)
    return C


## Closed form Black-Scholes formula for Put options
def EuPut_BlackScholes(S, K, rf, q, sigma, T):  # scheint ein fehler hier zu sein
    d_1 = (np.log(S / K) + (rf - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d_2 = (np.log(S / K) + (rf - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    phi = scs.norm.cdf(-d_1)
    P = K * np.exp(-rf * T) * scs.norm.cdf(-d_2) - S * np.exp(-q * T) * phi
    return P


## Bisection algorithm to find implied volatility
def BisecBSIV(PutCall, S, K, rf, q, T, a, b, MktPrice, Tol, MaxIter):
    if PutCall == 'C':
        lowCdif = MktPrice - EuCall_BlackScholes(S, K, rf, q, a, T)
        highCdif = MktPrice - EuCall_BlackScholes(S, K, rf, q, b, T)
    else:
        lowCdif = MktPrice - EuPut_BlackScholes(S, K, rf, q, a, T)
        highCdif = MktPrice - EuPut_BlackScholes(S, K, rf, q, b, T)
    # if an implied volatility cannot be found due to the model price not being outside the possible interval, denote as -1
    # and omit data point in the error metric calculation
    if lowCdif * highCdif > 0:
        y = -1
    else:
        for x in range(0, MaxIter):
            midP = (a + b) / 2
            if PutCall == 'C':
                midCdif = MktPrice - EuCall_BlackScholes(S, K, rf, q, midP, T)
            else:
                midCdif = MktPrice - EuPut_BlackScholes(S, K, rf, q, midP, T)
            # stop if difference is smaller than tolerance
            if abs(midCdif) < Tol:
                break
            else:
                if midCdif > 0:
                    # if midCdif is > 0 => BS price is lower than market price and thus volatility is lower and we adjust midP to be bigger
                    a = midP
                else:
                    # if midCdif is < 0 => BS price is higher than market price and thus volatility is higher and we adjust midP to be smaller
                    b = midP
        y = midP
    return y


## Heston model objective function that has to be minimized during calibration
def HestonObjFun(param, S, rf, q, MktPrice, K, T, PutCall, MktIV, x, w, trap, ObjFun, a, b, Tol, MaxIter, weights):
    kappa = param[0]
    theta = param[1]
    sigma = param[2]
    v0 = param[3]
    rho = param[4]
    lmbda = 0

    NK, NT = MktPrice.shape
    ModelPrice = np.empty([NK, NT])
    Vega = np.empty([NK, NT])
    error = np.empty([NK, NT])

    for k in range(0, NK):
        for t in range(0, NT):
            # Option pricing via Gauss-Laguerre Quadrature
            CallPrice = HestonPriceGaussLaguerre('C', S, K[k, t], T[t], rf, q, param, trap, x, w, IntMethod)
            if PutCall[k, t] == 'C':
                ModelPrice[k, t] = CallPrice
            else:
                ModelPrice[k, t] = CallPrice - S * np.exp(-q * T[t]) + np.exp(-rf * T[t]) * K[k, t]
            # MSE loss
            if ObjFun == 1:
                if weights == 0:  # baseline MSE
                    error[k, t] = (MktPrice[k, t] - ModelPrice[k, t]) ** 2
                elif weights == 1:  # MSE loss with absolute spread reciprocals
                    error[k, t] = (MktPrice[k, t] - ModelPrice[k, t]) ** 2 / abs(spread[k, t])
                elif weights == 2:  # MSE loss with squared spread reciprocals
                    error[k, t] = (MktPrice[k, t] - ModelPrice[k, t]) ** 2 / (spread[k, t] ** 2)
                elif weights == 3:  # MSE loss with square root spread reciprocals
                    error[k, t] = (MktPrice[k, t] - ModelPrice[k, t]) ** 2 / np.sqrt(spread[k, t])
                elif weights == 4:  # MSE loss with Vega weights
                    d = (np.log(S / K[k, t]) + (rf - q + 0.5 * MktIV[k, t] ** 2) * T[t]) / MktIV[k, t] / np.sqrt(T[t])
                    Vega[k, t] = S * scs.norm.pdf(d) * np.sqrt(T[t])
                    error[k, t] = (MktPrice[k, t] - ModelPrice[k, t]) ** 2 / Vega[k, t] ** 2
            # RMSE loss
            elif ObjFun == 2:
                error[k, t] = ((MktPrice[k, t] - ModelPrice[k, t]) ** 2) / MktPrice[k, t]
            # MSE loss using IV
            elif ObjFun == 3:
                ModelIV = BisecBSIV(PutCall[k, t], S, K[k, t], rf, q, T[t], a, b, ModelPrice[k, t], Tol, MaxIter)
                error[k, t] = (ModelIV - MktIV[k, t]) ** 2
    y = np.sum(error) / (NT * NK)
    return y


## Black-Scholes model objective function that has to be minimized during calibration
def BSObjFun(param, S, K, rf, q, T, MktPrice):
    sigma = param[0]
    NK, NT = MktPrice.shape
    ModelPrice = np.empty([NK, NT])
    error = np.empty([NK, NT])

    for k in range(0, NK):
        for t in range(0, NT):
            # Option pricing via Gauss-Laguerre Quadrature
            CallPrice = EuCall_BlackScholes(S, K[k, t], rf, q, sigma, T[t])
            if PutCall[k, t] == 'C':
                ModelPrice[k, t] = CallPrice
            else:
                ModelPrice[k, t] = CallPrice - S * np.exp(-q * T[t]) + np.exp(-rf * T[t]) * K[k, t]
            # MSE loss
            if ObjFun == 1 or 4:
                error[k, t] = (MktPrice[k, t] - ModelPrice[k, t]) ** 2
    z = np.sum(error) / (NT * NK)
    return z


## Function to calculate option prices under the Heston model using numerical integration, option prices with the closed-form BS model,
## implied volatility of the calculated Heston model prices and error statistics
def ErrorCalc(param, S, rf, q, MktPrice, K, T, PutCall, MktIV, x, w, trap, a, b, Tol, MaxIter):
    BSModelPrice = np.empty([NK, NT])
    ModelPrice = np.empty([NK, NT])
    NIModelIV = np.empty([NK, NT])
    MCModelIV = np.empty([NK, NT])
    NIerror = np.zeros((NK, NT))
    MCerror = np.zeros((NK, NT))
    SumNI = 0
    SumMC = 0

    for k in range(0, NK):
        for t in range(0, NT):
            # Heston model price with ITM call/put option values
            ModelPrice[k, t] = HestonPriceGaussLaguerre(PutCall[k, t], S, K[k, t], T[t], rf, q, param, trap, x, w,
                                                        IntMethod)
            # implied volatility of Heston model prices
            NIModelIV[k, t] = BisecBSIV(PutCall[k, t], S, K[k, t], rf, q, T[t], a, b, ModelPrice[k, t], Tol, MaxIter)
            MCModelIV[k, t] = BisecBSIV(PutCall[k, t], S, K[k, t], rf, q, T[t], a, b, MCPrice[k, t], Tol, MaxIter)
            # Black-Scholes model prices with first method (MSE loss minimization for sigma)
            BSModelPrice[k, t] = EuCall_BlackScholes(S, K[k, t], rf, q, resbs.x, T[t])
            # if there data points that have been marked during the Bisection algorithm, omit them here
            if NIModelIV[k, t] != -1:
                NIerror[k, t] = (NIModelIV[k, t] - MktIV[k, t]) ** 2
            else:
                SumNI += 1
            if MCModelIV[k, t] != -1:
                MCerror[k, t] = (MCModelIV[k, t] - MktIV[k, t]) ** 2
            else:
                SumMC += 1
    herror = (ModelPrice - MktPrice) ** 2
    BSerror = (MktPrice - BSModelPrice) ** 2
    # mean of IV squared error from Heston model
    NIErrorIV = np.sum(NIerror) / (NK * NT - SumNI)
    MCErrorIV = np.sum(MCerror) / (NK * NT - SumMC)
    # mean of price squared error from Heston model
    HPriceError = np.sum(herror) / (NK * NT)

    return NIModelIV, MCModelIV, ModelPrice, NIErrorIV, MCErrorIV, HPriceError, BSModelPrice, BSerror


## Euler, Milstein and Transformed Volatility simulation schemes
def EulerMilsteinSim(scheme, negvar, param, S, T, r, q, steps, N, alpha):
    dt = T / steps
    Stock = np.full(N, S)
    Vol = np.full(N, param[3])
    SVol = np.full(N, np.sqrt(param[3]))
    F = 0
    kappa = param[0]
    theta = param[1]
    sigma = param[2]
    rho = param[4]

    # Stock = np.zeros((steps+1, N))
    # Vol = np.zeros((steps+1, N))
    # SVol = np.zeros((steps+1, N))
    # Stock[0,:] = S
    # X[0,:] = np.log(S)
    # Vol[0,:] = param[3]
    # SVol[0,:] = np.sqrt(param[3])
    # Zv = np.random.normal(0, 1, (steps, N))
    # Zs = rho * Zv + np.sqrt(1 - rho ** 2) * np.random.normal(0, 1, (steps, N))

    m2 = ((sigma ** 2) / (4 * kappa)) * (1 - np.exp(-kappa * dt))

    for i in range(0, steps):
        if scheme == 'E':
            Vol[i + 1, :] = Vol[i, :] + kappa * (theta - Vol[i, :]) * dt + sigma * np.sqrt(Vol[i, :] * dt) * Zv[i, :]
        elif scheme == 'M':
            Vol[i + 1, :] = Vol[i, :] + kappa * (theta - Vol[i, :]) * dt + sigma * np.sqrt(Vol[i, :] * dt) * Zv[i,
                                                                                                             :] + 0.25 * sigma ** 2 * dt * (
                                        Zv[i, :] ** 2 - 1)
        elif scheme == 'IM':
            Vol[i + 1, :] = (Vol[i, :] + kappa * theta * dt + sigma * np.sqrt(Vol[i + 1, :] * dt) * Zv[i,
                                                                                                    :] + 0.25 * sigma ** 2 * dt * (
                                         Zv[i, :] ** 2 - 1)) / (1 + kappa * dt)
        elif scheme == 'WM':
            Vol[i + 1, :] = (Vol[i, :] + kappa * (theta - alpha * Vol[i, :]) * dt + sigma * np.sqrt(
                Vol[i, :] * dt) * Zv[i, :] + 0.25 * sigma ** 2 * dt * (Zv[i, :] ** 2 - 1)) / (
                                        1 + (1 - alpha) * kappa * dt)
        elif scheme == 'TV':
            m1 = theta + (Vol - theta) * np.exp(-kappa * dt)
            beta = np.sqrt(np.maximum(0, m1 - m2))
            thetav = (beta - SVol * np.exp(0.5 * -kappa * dt)) / (1 - np.exp(0.5 * -kappa * dt))
            Zv = np.random.normal(0, 1, N)
            Zs = rho * Zv + np.sqrt(1 - rho ** 2) * np.random.normal(0, 1, N)
            Stock = Stock * np.exp((r - q - 0.5 * Vol) * dt + SVol * np.sqrt(dt) * Zs)
            SVol = SVol + 0.5 * kappa * (thetav - SVol) * dt + 0.5 * sigma * np.sqrt(dt) * Zv
            Vol = SVol ** 2

            # m1 = theta + (Vol[i,:] - theta) * np.exp(-kappa * dt)
            # beta = np.sqrt(np.maximum(0,m1-m2))
            # thetav = (beta - SVol[i,:] * np.exp(0.5 * -kappa * dt)) / (1 - np.exp(0.5 * -kappa * dt))
            # SVol[i+1,:] = SVol[i,:] + 0.5 * kappa * (thetav - SVol[i,:]) * dt + 0.5 * sigma * np.sqrt(dt) * Zv[i,:]
            # Vol[i+1,:] = SVol[i+1,:]**2
        if scheme != 'TV':
            if np.any(Vol[i + 1, :] <= 0):
                F = F + np.count_nonzero(Vol[i + 1, :] <= 0)
                if negvar == 'R':
                    Vol[i + 1, :] = abs(Vol[i + 1, :])
                elif negvar == 'T':
                    Vol[i + 1, :] = np.maximum(0, Vol[i + 1, :])

        # Stock[i+1,:] = Stock[i,:] * np.exp((r - q - 0.5 * Vol[i,:]) * dt + SVol[i,:] * np.sqrt(dt) * Zs[i,:])
    return Stock, Vol, F


## Monte Carlo simulation
def EulerMilsteinPrice(scheme, negvar, param, PutCall, S0, K, T, r, q, steps, N, alpha):
    Stock, Vol, F = EulerMilsteinSim(scheme, negvar, param, S0, T, rf, q, steps, N, alpha)
    # ST = Stock[-1,:]
    SimPrice = np.empty(NK)

    for i in range(0, NK):
        if PutCall == 'C':
            SimPrice[i] = np.mean(np.maximum(Stock - K[i], 0))
        elif PutCall == 'P':
            SimPrice[i] = np.exp(-r * T) * np.mean(np.maximum(K[i] - Stock, 0)) + S * np.exp(-q * T) - K * np.exp(
                -r * T)

    return SimPrice  # , X, Vol, F


# function outputting the integrand for the density (for illustration purposes)
def HestonP(phi, kappa, theta, lmbda, rho, sigma, tau, lnSrange, S, r, q, v, Pnum, Trap):
    x = np.log(S)
    a = kappa * theta

    if Pnum == 1:
        u = 0.5
        b = kappa + lmbda - (rho * sigma)
    else:
        u = -0.5
        b = kappa + lmbda

    d = np.sqrt((rho * sigma * 1j * phi - b) ** 2 - sigma ** 2 * (2 * u * 1j * phi - phi ** 2))
    g = (b - rho * sigma * 1j * phi + d) / (b - rho * sigma * 1j * phi - d)

    if Trap == 1:
        # Little Heston trap formulation
        c = 1 / g
        D = (b - rho * sigma * 1j * phi - d) / sigma ** 2 * ((1 - np.exp(-d * tau)) / (1 - c * np.exp(-d * tau)))
        G = (1 - c * np.exp(-d * tau)) / (1 - c)
        C = (r - q) * 1j * phi * tau + a / sigma ** 2 * ((b - rho * sigma * 1j * phi - d) * tau - 2 * np.log(G))
    elif Trap == 0:
        G = (1 - g * np.exp(d * tau)) / (1 - g)
        C = (r - q) * 1j * phi * tau + a / sigma ** 2 * ((b - rho * sigma * 1j * phi + d) * tau - 2 * np.log(G))
        D = (b - rho * sigma * 1j * phi + d) / sigma ** 2 * ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau)))

    # characteristic function
    f = np.exp(C + D * v + 1j * phi * x)

    # Return real part of the integrand
    y = np.real(np.exp(-1j * phi * lnSrange) * f)

    return y


# constructing density (for illustration purposes)
def pdf_logstock(x, w, kappa, theta, lmbda, rho, sigma, T, S, lnSrange, rf, q, v0, Pnum, trap):
    # Numerical integration
    int2 = np.empty(len(x))
    PDF = np.empty(len(lnSrange))
    for i in range(0, len(lnSrange)):
        for k in range(0, len(x)):
            int2[k] = w[k] * HestonP(x[k], kappa, theta, lmbda, rho, sigma, T, lnSrange[i], S, rf, q, v0, Pnum,
                                     trap)
        PDF[i] = ((1 / math.pi) * np.sum(int2))
    return PDF, lnSrange


## simulating a single volatility process (for illustration purposes)
def simulation(scheme, negvar, param, S, T, r, q, steps, alpha):
    dt = T / steps
    F = 0
    kappa = param[0]
    theta = param[1]
    sigma = param[2]
    rho = param[4]

    Stock = np.zeros(steps + 1)
    Vol = np.zeros(steps + 1)
    SVol = np.zeros(steps + 1)
    Stock[0] = S
    Vol[0] = param[3]
    SVol[0] = np.sqrt(param[3])
    Zv = np.random.normal(0, 1, steps)
    Zs = rho * Zv + np.sqrt(1 - rho ** 2) * np.random.normal(0, 1, steps)

    m2 = ((sigma ** 2) / (4 * kappa)) * (1 - np.exp(-kappa * dt))

    for i in range(0, steps):
        if scheme == 'E':
            Vol[i + 1] = Vol[i] + kappa * (theta - Vol[i]) * dt + sigma * np.sqrt(Vol[i] * dt) * Zv[i]
        elif scheme == 'M':
            Vol[i + 1] = Vol[i] + kappa * (theta - Vol[i]) * dt + sigma * np.sqrt(Vol[i] * dt) * Zv[
                i] + 0.25 * sigma ** 2 * dt * (Zv[i] ** 2 - 1)
        elif scheme == 'IM':
            Vol[i + 1] = (Vol[i] + kappa * theta * dt + sigma * np.sqrt(Vol[i + 1] * dt) * Zv[
                i] + 0.25 * sigma ** 2 * dt * (Zv[i] ** 2 - 1)) / (1 + kappa * dt)
        elif scheme == 'WM':
            Vol[i + 1] = (Vol[i] + kappa * (theta - alpha * Vol[i]) * dt + sigma * np.sqrt(Vol[i] * dt) * Zv[
                i] + 0.25 * sigma ** 2 * dt * (Zv[i] ** 2 - 1)) / (1 + (1 - alpha) * kappa * dt)
        elif scheme == 'TV':

            m1 = theta + (Vol[i] - theta) * np.exp(-kappa * dt)
            beta = np.sqrt(np.maximum(0, m1 - m2))
            thetav = (beta - SVol[i] * np.exp(0.5 * -kappa * dt)) / (1 - np.exp(0.5 * -kappa * dt))
            SVol[i + 1] = SVol[i] + 0.5 * kappa * (thetav - SVol[i]) * dt + 0.5 * sigma * np.sqrt(dt) * Zv[i]
            Vol[i + 1] = SVol[i + 1] ** 2
        if scheme != 'TV':
            if np.any(Vol[i + 1] <= 0):
                F = F + np.count_nonzero(Vol[i + 1] <= 0)
                if negvar == 'R':
                    Vol[i + 1] = abs(Vol[i + 1])
                elif negvar == 'T':
                    Vol[i + 1] = np.maximum(0, Vol[i + 1])

        Stock[i + 1] = Stock[i] * np.exp((r - q - 0.5 * Vol[i]) * dt + SVol[i] * np.sqrt(dt) * Zs[i])
    return Stock, Vol, F


## Methods
trap = 1  # characteristic function form. 1 = Albrecher's formulation, 0 = Heston's original formulation
IntMethod = 2  # integral method. 1 = calculation with two integrals, 2 = calculation using one integral
ObjFun = 1  # loss function method. 1 = MSE, 2 = Relative MSE, 3 = IV
weights = 2  # weighting method. 0 = no weights, 1 = absolute spread reciprocals, 2 = squared spread reciprocals, 3 = square root spread reciprocals, 4 = BS Vega reciprocals

## Data and parameters
options = pd.read_excel('1yopt.xlsx')  # import DAX option chain from November 3rd, 2020
# time to maturity
T = options['Expiry'].to_numpy() / 365
T = np.unique(T)
# strike prices
K = options['Strike'].to_numpy()
K = np.reshape(K, (int(len(K) / len(T)), len(T)), 'F')
NK, NT = K.shape
MktIV = np.empty([NK, NT])

## other option data
rf = 0
q = 0
S = 11994.04

# Call/Put switch mechanism to always have ITM options
# PutCall = S - K
# PutCall = np.where(PutCall < 0, 'P', 'C')
PutCall = matlib.repmat('C', NK,
                        NT)  # matrix that will define each option as either call or put. in this case we are using calls only
# ask, bid, mid matrices
mid = ((options['Bid'] + options['Ask']) / 2).to_numpy()
MktPrice = np.reshape(mid, (int(len(mid) / len(T)), len(T)), 'F')
bid = options['Bid'].to_numpy()
bid = np.reshape(bid, (int(len(bid) / len(T)), len(T)), 'F')
ask = options['Ask'].to_numpy()
ask = np.reshape(ask, (int(len(ask) / len(T)), len(T)), 'F')
spread = ask - bid
## initial values for minimization
# starting values for Heston model minimization
x0 = np.array([3, 0.5, 0.5, 0.5, -0.5])
# starting value for Black-Scholes model minimization
x00 = 0.2

# bisection parameters
a = 0.001
b = 3
Tol = 0.0000001
MaxIter = 1000

# bounds for local optimization
param = np.empty(5)
lmbda = 0
e = 0.00001
bnds = [(e, 10), (e, 4), (e, 4), (e, 4), (-0.999, 0.999)]
bnds1 = [(e, 20)]

if __name__ == "__main__":
    ## Computation

    # Gauss-Laguerre weights and abscissas
    x, w = GenerateGaussLaguerre(n)

    for k in range(0, NK):
        for t in range(0, NT):
            MktIV[k, t] = BisecBSIV(PutCall[k, t], S, K[k, t], rf, q, T[t], a, b, MktPrice[k, t], Tol, MaxIter)

    startc = time.time()
    # local minimization
    # parallel computation of L-BFGS-B algorithm using optimparallel package by Florian Gerber [Heston model]
    res = minimize_parallel(HestonObjFun, x0, args=(
    S, rf, q, MktPrice, K, T, PutCall, MktIV, x, w, trap, ObjFun, a, b, Tol, MaxIter, weights,), bounds=bnds,
                            parallel={'loginfo': True, 'time': True})

    # L-BFGS-B algorithm using scipy package [Heston model]
    # res = minimize(HestonObjFun, x0, method='L-BFGS-B', args=(S, rf, q, MktPrice, K, T, PutCall, MktIV, x, w, trap, ObjFun, a, b, Tol, MaxIter, weights,),bounds=bnds)

    # Sequential Least Squares Programming algorithm using scipy package [Heston model]
    # res = minimize(HestonObjFun, x0, method='SLSQP', args=(S, rf, q, MktPrice, K, T, PutCall, MktIV, x, w, trap, ObjFun, a, b, Tol, MaxIter, weights,),bounds=bnds)

    # Sequential Least Squares Programming using scipy package [Black-Scholes model]
    # resbs = minimize(BSObjFun, x00, method='SLSQP', args=(S,K,rf,q,T,MktPrice,),bounds=bnds1)

    # global minimization
    # parallel computation of Differential Evolution 'best1bin' algorithm using scipy package [Heston model], workers = number of cores used. -1 = all cores
    # res = differential_evolution(HestonObjFun,bnds,(S,rf,q,MktPrice,K,T,PutCall,MktIV,x,w,trap,ObjFun,a,b,Tol,MaxIter, weights,), updating='deferred', workers=-1)

    # Dual Annealing using scipy package [Heston model]
    # res = dual_annealing(HestonObjFun, bounds=bnds, args=(S,rf,q,MktPrice,K,T,PutCall,MktIV,x,w,trap,ObjFun,a,b,Tol,MaxIter, weights,))

    # Differential Evolution 'best1bin' using scipy package [Black-Scholes model]
    resbs = differential_evolution(BSObjFun, bnds1, (S, K, rf, q, T, MktPrice,))
    end_time = time.time()
    print(f'Execution time for calibration: {time.time() - startc} seconds')

    # assigning calibrated parameters to param vector and individual global variables
    param = res.x
    kappa, theta, sigma, v0, rho = param

    # simulation parameters (large numbers might crash the machine)
    steps = 5000  # steps of discretization
    N = 50000  # number of stock price paths
    schemeV = 'TV'  # simulation scheme
    negvar = 'T'  # variance transformation method
    alpha = 0.5  # coefficient for weighted Milstein

    # Parallel computation of MC simulation (adjust step size and number of simulations based on the performance of your PC)
    input_list = [(schemeV, negvar, param, 'C', S, K[:, t], T[t], rf, q, steps, N, alpha) for t in range(0, NT)]
    startmc = time.time()
    pool = Pool()
    PMC = pool.starmap(EulerMilsteinPrice, input_list)
    pool.close()
    end_time = time.time()
    print(f'Execution time for MC option pricing: {time.time() - startmc} seconds')
    MCPrice = np.asarray(PMC).transpose()

    # Single core computation of MC simulation (use this on less powerful machines)
    '''
    MCPrice = np.empty([NK,NT])
    startmc1 = time.time()
    for t in range(0, NT):
        MCPrice[:, t] = EulerMilsteinPrice(schemeV, negvar, param, 'C', S, K[:, t], T[t], rf, q, steps, N, alpha)
    end_time = time.time()
    print(f'Execution time for MC option pricing: {time.time() - startmc1} seconds')
    '''

    # computing model implied volatility, model price, MSE of model implied volatility, MSE of price, BS model price
    NIModelIV, MCModelIV, ModelPrice, NIErrorIV, MCErrorIV, HPriceError, BSModelPrice, BSerror = ErrorCalc(param, S, rf,
                                                                                                           q, MktPrice,
                                                                                                           K, T,
                                                                                                           PutCall,
                                                                                                           MktIV, x, w,
                                                                                                           trap, a, b,
                                                                                                           Tol, MaxIter)

    # starto = time.time()
    # Black-Scholes model prices calculation using calibrated sigma^2 and Monte Carlo simulation of option prices
    # for k in range(0, NK):
    # for t in range(0, NT):
    # MCPrice[k,t] = EulerMilsteinPrice(schemeV, negvar, param, 'C', S, K[k,t], T[t], rf, q, steps, N, alpha)[0]
    # end_time = time.time()
    # print(f'Execution time for BS option pricing: {time.time() - starto} seconds')

    # more error statistics
    PriceError = np.sum(BSerror) / (NK * NT)  # MSE of BS price
    HestonError = abs(ModelPrice - MktPrice)
    withinspread = HestonError < (
                ask - bid)  # check if absolute difference of option prices is greater than ask-bid spread
    RelBSError = abs((MktPrice - BSModelPrice) / MktPrice)
    MRelBSError = np.sum(RelBSError) / (NK * NT)  # Relative Mean Absolute Error BS model
    RelHestonError = abs((MktPrice - ModelPrice) / MktPrice)
    MRealHestonError = np.sum(RelHestonError) / (NK * NT)  # Relative Mean Absolute Error Heston model
    MCRealHestonError = abs((MktPrice - MCPrice) / MktPrice)
    MMCRealHestonerror = np.sum(MCRealHestonError) / (
                NK * NT)  # Relative Mean Absolute Error Monte Carlo simulation under Heston model

    # simulation of stochastic process (for illustration purposes)
    StockSim, VolSim, F = simulation('TV', negvar, param, S, T[5], rf, q, 100000, alpha)
    lnSrange = np.linspace(np.log(S) - 1, np.log(S) + 1, 2000)

    # density (for illustration purposes)
    PDF_0, lnSrange = pdf_logstock(x, w, kappa, theta, lmbda, 0, sigma, T[5], S, lnSrange, rf, q, v0, 2, trap)
    PDF_0sigma = pdf_logstock(x, w, kappa, theta, lmbda, 0, 0.05, T[5], S, lnSrange, rf, q, v0, 2, trap)[0]
    PDF_negrho = pdf_logstock(x, w, kappa, theta, lmbda, rho, sigma, T[5], S, lnSrange, rf, q, v0, 2, trap)[0]
    PDF_posrho = pdf_logstock(x, w, kappa, theta, lmbda, -rho, sigma, T[5], S, lnSrange, rf, q, v0, 2, trap)[0]

    ## Console output summary
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="dim", justify="center", width=18)
    table.add_column("Parameters", justify="center", width=20)
    table.add_column("Parameter values", justify="center", width=20)
    table.add_column("MSE IV", justify="center", width=18)
    table.add_column("MSE Price", justify="center", width=18)
    table.add_column("RMAE Price %", justify="center", width=18)
    table.add_row("Heston Model", "kappa, theta, sigma, v0, rho", str(param), str(NIErrorIV), str(HPriceError),
                  str(MRealHestonError * 100))
    table.add_row("BS-Model", "sigma", str(resbs.x), "none", str(PriceError), str(MRelBSError * 100))
    console.print(table)

    ## Interactive line and surface plot
    scope = PlotlyScope()
    config = {'displaylogo': False, 'toImageButtonOptions': {'format': 'svg'}}

    fig = make_subplots(rows=3, cols=2, start_cell="top-left", subplot_titles=(
    "Maturity 45 days", "Maturity 73 days", "Maturity 136 days", "Maturity 227 days", "Maturity 318 days",
    "Maturity 409 days"))
    fig.add_trace(
        go.Scatter(x=K[:, 0], y=MktIV[:, 0], name='Market IV', line=dict(color='royalblue'), legendgroup="mktiv"),
        row=1, col=1, )
    fig.add_trace(go.Scatter(x=K[:, 0], y=NIModelIV[:, 0], name='Model IV', line=dict(color='firebrick', dash='dot'),
                             legendgroup="modeliv"), row=1, col=1)
    fig.add_trace(go.Scatter(x=K[:, 0], y=MCModelIV[:, 0], name='MC Model IV', line=dict(color='limegreen', dash='dot'),
                             legendgroup="mcmodeliv"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=K[:, 0], y=MktIV[:, 1], name='Market IV', line=dict(color='royalblue'), legendgroup="mktiv",
                   showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=K[:, 0], y=NIModelIV[:, 1], name='Model IV', line=dict(color='firebrick', dash='dot'),
                             legendgroup="modeliv", showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=K[:, 0], y=MCModelIV[:, 1], name='MC Model IV', line=dict(color='limegreen', dash='dot'),
                             legendgroup="mcmodeliv", showlegend=False), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=K[:, 0], y=MktIV[:, 2], name='Market IV', line=dict(color='royalblue'), legendgroup="mktiv",
                   showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=K[:, 0], y=NIModelIV[:, 2], name='Model IV', line=dict(color='firebrick', dash='dot'),
                             legendgroup="modeliv", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=K[:, 0], y=MCModelIV[:, 2], name='MC Model IV', line=dict(color='limegreen', dash='dot'),
                             legendgroup="mcmodeliv", showlegend=False), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=K[:, 0], y=MktIV[:, 3], name='Market IV', line=dict(color='royalblue'), legendgroup="mktiv",
                   showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=K[:, 0], y=NIModelIV[:, 3], name='Model IV', line=dict(color='firebrick', dash='dot'),
                             legendgroup="modeliv", showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=K[:, 0], y=MCModelIV[:, 3], name='MC Model IV', line=dict(color='limegreen', dash='dot'),
                             legendgroup="mcmodeliv", showlegend=False), row=2, col=2)
    fig.add_trace(
        go.Scatter(x=K[:, 0], y=MktIV[:, 4], name='Market IV', line=dict(color='royalblue'), legendgroup="mktiv",
                   showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=K[:, 0], y=NIModelIV[:, 4], name='Model IV', line=dict(color='firebrick', dash='dot'),
                             legendgroup="modeliv", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=K[:, 0], y=MCModelIV[:, 4], name='MC Model IV', line=dict(color='limegreen', dash='dot'),
                             legendgroup="mcmodeliv", showlegend=False), row=3, col=1)
    fig.add_trace(
        go.Scatter(x=K[:, 0], y=MktIV[:, 5], name='Market IV', line=dict(color='royalblue'), legendgroup="mktiv",
                   showlegend=False), row=3, col=2)
    fig.add_trace(go.Scatter(x=K[:, 0], y=NIModelIV[:, 5], name='Model IV', line=dict(color='firebrick', dash='dot'),
                             legendgroup="modeliv", showlegend=False), row=3, col=2)
    fig.add_trace(go.Scatter(x=K[:, 0], y=MCModelIV[:, 5], name='MC Model IV', line=dict(color='limegreen', dash='dot'),
                             legendgroup="mcmodeliv", showlegend=False), row=3, col=2)
    fig.show(config=config)
    # fig.write_image("output/O1_W2/MSE_L-BFGS_O1_W2/fig.pdf")
    # fig.write_html("output/O1_W2/MSE_L-BFGS_O1_W2/fig.html", include_plotlyjs=False, full_html=True)

    fig1 = make_subplots(rows=1, cols=2, start_cell="top-left", specs=[[{'is_3d': True}, {'is_3d': True}]],
                         horizontal_spacing=0)
    fig1.add_trace(
        go.Surface(z=NIModelIV, x=T, y=K[:, 0], coloraxis="coloraxis", showlegend=True, name='Model IV', opacity=0.6,
                   hidesurface=False, contours={"x": {"show": True, "usecolormap": False, "color": "black"},
                                                "y": {"show": True, "usecolormap": False, "color": "black"}}), row=1,
        col=1)
    fig1.add_trace(go.Surface(z=MktIV, x=T, y=K[:, 0], coloraxis="coloraxis", colorscale='viridis', showlegend=True,
                              name='Market IV', opacity=0.5, hidesurface=False,
                              contours={"x": {"show": True, "usecolormap": False, "color": "black"},
                                        "y": {"show": True, "usecolormap": False, "color": "black"}}), row=1, col=1)
    fig1.add_trace(go.Scatter3d(y=np.tile(T, 58), x=K.flatten('F'), z=RelHestonError.flatten('F') * 100, mode='markers',
                                name="Option Prices from Numerical Integration", marker=dict(size=3, opacity=0.8)),
                   row=1, col=2)
    fig1.add_trace(
        go.Scatter3d(y=np.tile(T, 58), x=K.flatten('F'), z=MCRealHestonError.flatten('F') * 100, mode='markers',
                     name="Option Prices from MC Simulation", marker=dict(size=3, opacity=0.8)), row=1, col=2)
    fig1.update_layout(scene=dict(
        xaxis=dict(
            ticktext=['45', '73', '136', '227', '318', '409'],
            tickvals=[0.12328767, 0.2, 0.37260274, 0.62191781, 0.87123288, 1.12054795]),
        xaxis_title="Maturity",
        yaxis_title="Strike price",
        zaxis_title="Implied Volatility"),
        legend=dict(yanchor="bottom", y=0.99, xanchor="left", x=0.01, orientation="h"),
        margin=dict(r=0, b=0, l=0, t=0),
        template='plotly_white',
        autosize=False,
        width=1000,
        font=dict(size=10),
        coloraxis_colorbar=dict(len=0.9, title="Implied Volatility", x=-0.1)
        # coloraxis= {'colorscale':'viridis'}
    )
    fig1.update_layout(scene2=dict(
        yaxis=dict(
            ticktext=['45', '73', '136', '227', '318', '409'],
            tickvals=[0.12328767, 0.2, 0.37260274, 0.62191781, 0.87123288, 1.12054795]),
        yaxis_title="Maturity",
        xaxis_title="Strike price",
        zaxis_title="Absolute Percentage Error"),
        legend=dict(yanchor="bottom", y=0.99, xanchor="left", x=0.01, orientation="h"),
        margin=dict(r=0, b=0, l=0, t=0),
        template='plotly_white',
        autosize=False,
        width=1000,
        font=dict(size=10),
        coloraxis_colorbar=dict(len=1, title="Implied Volatility", x=-0.1)
        # coloraxis= {'colorscale':'viridis'}
    )
    # with open("figure1.png", "wb") as f:
    # f.write(scope.transform(fig1, format="png"))
    fig1.update_scenes(camera_eye=dict(x=1.5, y=1.5, z=1.65))
    fig1.show(config=config)
    # fig1.write_image("output/O1_W2/MSE_L-BFGS_O1_W2/fig1.pdf")
    # fig1.write_html("output/O1_W2/MSE_L-BFGS_O1_W2/fig1.html", include_plotlyjs=False, full_html=True)

    fig2 = make_subplots(rows=3, cols=2, start_cell="top-left", subplot_titles=(
        "Maturity 45 days", "Maturity 73 days", "Maturity 136 days", "Maturity 227 days", "Maturity 318 days",
        "Maturity 409 days"))
    fig2.add_trace(
        go.Scatter(x=K[:, 0], y=MktPrice[:, 0], name='Market Price', line=dict(color='royalblue'),
                   legendgroup="mktprice"),
        row=1, col=1, )
    fig2.add_trace(
        go.Scatter(x=K[:, 0], y=ModelPrice[:, 0], name='Model Price', line=dict(color='firebrick', dash='dot'),
                   legendgroup="modelprice"), row=1, col=1)
    fig2.add_trace(go.Scatter(x=K[:, 0], y=MCPrice[:, 0], name='MC Model Price', line=dict(color='indigo', dash='dot'),
                              legendgroup="mcmodelprice"), row=1, col=1)
    fig2.add_trace(
        go.Scatter(x=K[:, 0], y=MktPrice[:, 1], name='Market Price', line=dict(color='royalblue'),
                   legendgroup="mktprice",
                   showlegend=False), row=1, col=2)
    fig2.add_trace(
        go.Scatter(x=K[:, 0], y=ModelPrice[:, 1], name='Model Price', line=dict(color='firebrick', dash='dot'),
                   legendgroup="modelprice", showlegend=False), row=1, col=2)
    fig2.add_trace(go.Scatter(x=K[:, 0], y=MCPrice[:, 1], name='MC Model Price', line=dict(color='indigo', dash='dot'),
                              legendgroup="mcmodelprice"), row=1, col=2)
    fig2.add_trace(
        go.Scatter(x=K[:, 0], y=MktPrice[:, 2], name='Market Price', line=dict(color='royalblue'),
                   legendgroup="mktprice",
                   showlegend=False), row=2, col=1)
    fig2.add_trace(
        go.Scatter(x=K[:, 0], y=ModelPrice[:, 2], name='Model Price', line=dict(color='firebrick', dash='dot'),
                   legendgroup="modelprice", showlegend=False), row=2, col=1)
    fig2.add_trace(go.Scatter(x=K[:, 0], y=MCPrice[:, 3], name='MC Model Price', line=dict(color='indigo', dash='dot'),
                              legendgroup="mcmodelprice"), row=2, col=1)
    fig2.add_trace(
        go.Scatter(x=K[:, 0], y=MktPrice[:, 3], name='Market Price', line=dict(color='royalblue'),
                   legendgroup="mktprice",
                   showlegend=False), row=2, col=2)
    fig2.add_trace(
        go.Scatter(x=K[:, 0], y=ModelPrice[:, 3], name='Model Price', line=dict(color='firebrick', dash='dot'),
                   legendgroup="modelprice", showlegend=False), row=2, col=2)
    fig2.add_trace(go.Scatter(x=K[:, 0], y=MCPrice[:, 3], name='MC Model Price', line=dict(color='indigo', dash='dot'),
                              legendgroup="mcmodelprice"), row=2, col=2)
    fig2.add_trace(
        go.Scatter(x=K[:, 0], y=MktPrice[:, 4], name='Market Price', line=dict(color='royalblue'),
                   legendgroup="mktprice",
                   showlegend=False), row=3, col=1)
    fig2.add_trace(
        go.Scatter(x=K[:, 0], y=ModelPrice[:, 4], name='Model Price', line=dict(color='firebrick', dash='dot'),
                   legendgroup="modelprice", showlegend=False), row=3, col=1)
    fig2.add_trace(go.Scatter(x=K[:, 0], y=MCPrice[:, 4], name='MC Model Price', line=dict(color='indigo', dash='dot'),
                              legendgroup="mcmodelprice"), row=3, col=1)
    fig2.add_trace(
        go.Scatter(x=K[:, 0], y=MktPrice[:, 5], name='Market Price', line=dict(color='royalblue'),
                   legendgroup="mktprice",
                   showlegend=False), row=3, col=2)
    fig2.add_trace(
        go.Scatter(x=K[:, 0], y=ModelPrice[:, 5], name='Model Price', line=dict(color='firebrick', dash='dot'),
                   legendgroup="modelprice", showlegend=False), row=3, col=2)
    fig2.add_trace(go.Scatter(x=K[:, 0], y=MCPrice[:, 5], name='MC Model Price', line=dict(color='indigo', dash='dot'),
                              legendgroup="mcmodelprice"), row=3, col=2)
    fig2.show(config=config)
    # fig2.write_image("output/O1_W2/MSE_L-BFGS_O1_W2/fig2.pdf")
    # fig2.write_html("output/O1_W2/MSE_L-BFGS_O1_W2/fig2.html", include_plotlyjs=False, full_html=True)

    # np.savetxt('output/O1_W2/MSE_L-BFGS_O1_W2/HestonError.csv', HestonError, delimiter=',')
    # np.savetxt('output/O1_W2/MSE_L-BFGS_O1_W2/MCPrice.csv', MCPrice, delimiter=',')
    # np.savetxt('output/O1_W2/MSE_L-BFGS_O1_W2/MCRealHestonError.csv', MCRealHestonError, delimiter=',')
    # np.savetxt('output/O1_W2/MSE_L-BFGS_O1_W2/NIModelIV.csv', NIModelIV, delimiter=',')
    # np.savetxt('output/O1_W2/MSE_L-BFGS_O1_W2/MCModelIV.csv', MCModelIV, delimiter=',')
    # np.savetxt('output/O1_W2/MSE_L-BFGS_O1_W2/ModelPrice.csv', ModelPrice, delimiter=',')
    # np.savetxt('output/O1_W2/MSE_L-BFGS_O1_W2/BSModelPrice.csv', BSModelPrice, delimiter=',')
    # np.savetxt('output/O1_W2/MSE_L-BFGS_O1_W2/RelHestonError.csv', RelHestonError, delimiter=',')
    # np.savetxt('output/O1_W2/MSE_L-BFGS_O1_W2/param.csv', param, delimiter=',')
    # np.savetxt('output/O1_W2/MSE_L-BFGS_O1_W2/OtherMeasures.csv', (NIErrorIV,MCErrorIV, HPriceError, MMCRealHestonerror, MRealHestonError), header="ErrorIV MCErrorIV HPriceError MMCRealHestonError MRealHestonError", delimiter=',')

    fig3 = make_subplots(rows=1, cols=2, start_cell="top-left")
    fig3.add_trace(go.Scatter(x=lnSrange, y=PDF_0sigma, mode='lines', name='ρ = 0, σ = 0'), row=1, col=1)
    fig3.add_trace(go.Scatter(x=lnSrange, y=PDF_0, mode='lines', name='ρ = 0'), row=1, col=1)
    fig3.add_trace(go.Scatter(x=lnSrange, y=PDF_negrho, mode='lines', name='ρ = -0.725', line=dict(dash='dash')), row=1,
                   col=1)
    fig3.add_trace(go.Scatter(x=lnSrange, y=PDF_posrho, mode='lines', name='ρ = 0.725', line=dict(dash='dot')), row=1,
                   col=1)
    fig3.add_trace(
        go.Scatter(x=np.linspace(0, T[5], 10000 + 1), y=np.sqrt(VolSim), name='Variance Process', mode='lines',
                   showlegend=False), row=1, col=2)
    fig3.update_xaxes(title_text="Log Stock Price", row=1, col=1)
    fig3.update_xaxes(title_text="Time to Maturity", row=1, col=2)
    fig3.update_yaxes(title_text="Variance", row=1, col=2)
    fig3.update_layout(template='plotly_white',
                       legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.3, font=dict(size=10)), margin=dict(b=0),
                       autosize=True)
    fig3.show(config=config)
    # fig3.write_image("output/O1_W2/MSE_L-BFGS_O1_W2/fig3.pdf")
    # fig3.write_html("output/O1_W2/MSE_L-BFGS_O1_W2/fig3.html", include_plotlyjs=False, full_html=True)
