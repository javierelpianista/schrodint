'''
This is a basic implementation of the explicit RK4 method.
It is the 4th order Runge-Kutta method with tabularized coefficients.
'''

import numpy as np
import matplotlib.pyplot as plt

neq = 2

def f(t, y):
    return np.array([y[1], (t**2-9)*y[0]])

def exact(t):
    return (4*t**4-12*t**2+3)/3*np.exp(-t**2/2)

# step size
h = 0.01

# initial values
y0 = np.array([1, 0])
t0 = 0
C = np.array([0, 1/2, 1/2, 1])
B = np.array([1/6, 1/3, 1/3, 1/6])
A = np.array([
        [1/2, 0, 0],
        [0, 1/2, 0],
        [0, 0, 1]
        ])
k = np.zeros([len(C), neq])
k.fill(np.inf)

# step size
h = 0.01

# initial values
y0 = np.array([1, 0])
t0 = 0

y = y0
t = t0

X = [t]
Y = [y]

print(B.shape)

while t <= 6:
    for i in range(len(C)):
        par = A[i-1,:i]@k[:i, :]

        k[i,:] = f(t+C[i]*h, y+par*h)

    y = y + h*B@k #np.dot(B, k)
    t = t + h

    #print('{:.14e}'.format(y))
   
    X.append(t)
    Y.append(y)

X = np.array(X)
Y = np.array(Y)

plt.plot(X, Y[:,0], '.', label = 'approx')
plt.plot(X, exact(X), label = 'exact')
plt.legend()
plt.show()
