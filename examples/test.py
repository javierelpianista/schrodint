import numpy as np
import matplotlib.pyplot as plt
from rkintegrator import RKIntegrator

def fun(x):
    return -2+2*x[0]+x[1]

def exact(x):
    return np.exp(x) - 2*x

x0 = [0, 1]

r1 = RKIntegrator.rk4(fun, x0)
r2 = RKIntegrator.three_eight_rule(fun, x0)
r3 = RKIntegrator.midpoint(fun, x0)

xmax = 2

XX = []

for n, inte in enumerate((r1, r2, r3)):
    inte.step_size = 0.05
    X = []

    while inte.x[0] < xmax:
        X.append(next(inte))

    X = np.array(X)

    plt.plot(X[:,0], X[:,1], '.', label = str(n))

    XX.append(X[:,1])

print(XX[0]-XX[1])
print(XX[0]-XX[2])

X = np.linspace(0,xmax, 1000)
plt.plot(X, exact(X))
plt.legend()
plt.show()
