'''
This is a basic implementation of the explicit RK4 method.
It is the 4th order Runge-Kutta method with tabularized coefficients.
'''

import numpy as np
import matplotlib.pyplot as plt

def f(t, y):
    return y*t

def exact(t):
    return np.exp(t**2/2)

C = np.array([0, 1/2, 1/2, 1])
B = np.array([1/6, 1/3, 1/3, 1/6])
A = np.array([
        [1/2, 0, 0],
        [0, 1/2, 0],
        [0, 0, 1]
        ])
k = np.zeros(len(C))
k.fill(np.inf)

# step size
h = 0.5

# initial values
y0 = 1
t0 = 0

y = y0
t = t0

X = [t]
Y = [y]

while t <= 10:
    for i in range(len(C)):
        par = 0
        par += np.dot(A[i-1,:i], k[:i])

        k[i] = f(t+C[i]*h, y+par*h)

    y += h*np.dot(B, k)
    t = t + h

    print('{:.14e}'.format(y))
   
    X.append(t)
    Y.append(y)

X = np.array(X)

plt.plot(X, Y, '.', label = 'approx')
plt.plot(X, exact(X), label = 'exact')
plt.legend()
plt.show()
