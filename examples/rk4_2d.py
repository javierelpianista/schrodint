'''
This is a basic implementation of the RK4 method for a second order 
differential equation.
'''

import numpy as np
import matplotlib.pyplot as plt

def f(t, y):
    return np.array([y[1], (t**2-5)*y[0]])

def exact(t):
    return (1-2*t**2)*np.exp(-t**2/2)

# step size
h = 0.01

# initial values
y0 = np.array([1, 0])
t0 = 0

y = y0
t = t0

X = [t]
Y = [y]

while t <= 6:
    k1 = f(t, y)
    k2 = f(t+h/2, y+h*k1/2)
    k3 = f(t+h/2, y+h*k2/2)
    k4 = f(t+h, y+h*k3)

    y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
    t = t + h

    X.append(t)
    Y.append(y)

X = np.array(X)
Y = np.array(Y)

plt.plot(X, Y[:,0], '.', label = 'approx0')
#plt.plot(X, Y[:,1], '.', label = 'approx1')
plt.plot(X, exact(X), label = 'exact')
plt.legend()
plt.show()
