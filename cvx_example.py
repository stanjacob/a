# Generate data for Huber regression.
import numpy as np
np.random.seed(1)
n = 300
SAMPLES = int(1.5*n)
beta_true = 5*np.random.normal(size=(n,1))
X = np.random.randn(n, SAMPLES)
Y = np.zeros((SAMPLES,1))
v = np.random.normal(size=(SAMPLES,1))

# Generate data for different values of p.
# Solve the resulting problems.
# WARNING this script takes a few minutes to run.
from cvxpy import *
TESTS = 50
lsq_data = np.zeros(TESTS)
huber_data = np.zeros(TESTS)
prescient_data = np.zeros(TESTS)
p_vals = np.linspace(0,0.15, num=TESTS)
for idx, p in enumerate(p_vals):
    # Generate the sign changes.
    factor = 2*np.random.binomial(1, 1-p, size=(SAMPLES,1)) - 1
    Y = factor*X.T.dot(beta_true) + v
    
    # Form and solve a standard regression problem.
    beta = Variable(n)
    fit = norm(beta - beta_true)/norm(beta_true)
    cost = norm(X.T*beta - Y)
    prob = Problem(Minimize(cost))
    prob.solve()
    lsq_data[idx] = fit.value
    
    # Form and solve a prescient regression problem,
    # i.e., where the sign changes are known.
    cost = norm(mul_elemwise(factor, X.T*beta) - Y)
    Problem(Minimize(cost)).solve()
    prescient_data[idx] = fit.value
    
    # Form and solve the Huber regression problem.
    cost = sum_entries(huber(X.T*beta - Y, 1))
    Problem(Minimize(cost)).solve()
    huber_data[idx] = fit.value