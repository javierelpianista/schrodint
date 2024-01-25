'''
This is a basic implementation of the adaptive Runge-Kutta-Fehlberg method
with the original coefficients.
'''

import numpy as np
import matplotlib.pyplot as plt

def f(t, y):
    return np.sin(t)**2*y

def exact(t):
    return np.exp(-1/4*np.sin(2*t)+1/2*t)

C = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
B = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
Bp = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])
A = np.array([
        [1/4, 0, 0, 0, 0],
        [3/32, 9/32, 0, 0 ,0],
        [1932/2197, -7200/2197, 7296/2197, 0, 0],
        [439/216, -8, 3680/513, -845/4104, 0],
        [-8/27, 2, -3544/2565, 1859/4104, -11/40]
        ])

k = np.zeros(len(C))
k.fill(np.inf)

# step size
h = 5

# initial values
y0 = 1
t0 = 0

y = y0
t = t0

X = [t]
Y = [y]

while t <= 100:
    for i in range(len(C)):
        par = 0
        par += np.dot(A[i-1,:i], k[:i])

        k[i] = f(t+C[i]*h, y+par*h)

    dy = h*np.dot(B, k)
    dyp = h*np.dot(Bp, k)

    if np.isnan(dy):
        print('nan at y = ', y)
        break
    app_err = dy - dyp
    app_rel_err = abs(app_err/(y+dy))

    #err = exact(t) - y
    #rel_err = abs(err/exact(t))

    if app_rel_err > 1e-5:
        h = h*0.8
        continue
    elif app_rel_err < 1e-10:
        h = h*1.2

    print('{:.14e}{:14.3e}{:12.3e}{:12.3e}'.format(y, app_err, app_rel_err, h))

    t = t + h
    y = y + dy
    yp = y + dyp
   
    X.append(t)
    Y.append(y)

X = np.array(X)

plt.plot(X, Y, '.', label = 'approx')
plt.plot(X, exact(X), label = 'exact')
plt.legend()
plt.show()
