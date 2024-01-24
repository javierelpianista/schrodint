'''
This is a basic implementation of the RK4 method.
'''

import numpy as np
import matplotlib.pyplot as plt

def f(t, y):
    return y*t

def exact(t):
    return np.exp(t**2/2)

# step size
h = 0.05

# initial values
y0 = 1
t0 = 0

y = y0
t = t0

X = [t]
Y = [y]

while t <= 10:
    k1 = f(t, y)
    k2 = f(t+h/2, y+h*k1/2)
    k3 = f(t+h/2, y+h*k2/2)
    k4 = f(t+h, y+h*k3)

    y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
    t = t + h

    print('{:.14e}'.format(y))
   
    X.append(t)
    Y.append(y)

X = np.array(X)

plt.plot(X, Y, '.', label = 'approx')
plt.plot(X, exact(X), label = 'exact')
plt.legend()
plt.show()
