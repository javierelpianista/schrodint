import numpy as np

class RKIntegrator:

    def __init__(self, cv, bv, rkm, fun, x0):

        self.cv = cv
        self.bv = bv
        self.rkm = rkm

        self.fun = fun
        self.x = x0

        n, m = self.rkm.shape

        self.nsteps = n+1

        if n != m:
            raise ValueError('rkm is not a square matrix')

        if not np.allclose(np.tril(rkm), rkm):
            raise ValueError('rkm is not a lower triangular matrix')

        if n != len(self.cv)-1 or n != len(self.bv)-1:
            raise ValueError('Wrong dimensions in cv, bv, and rkm')

        self.step_size = 0.05

    @classmethod 
    def midpoint(cls, fun, x0):
        rkm = np.array([[1/2]])
        cv = np.array([0, 1/2])
        bv = np.array([0, 1])

        return cls(cv, bv, rkm, fun, x0)

    @classmethod
    def rk4(cls, fun, x0):
        rkm = np.array([[1/2, 0  , 0], 
                        [0  , 1/2, 0],
                        [0  , 0  , 1]])

        cv = np.array([0, 1/2, 1/2, 1])
        bv = np.array([1/6, 1/3, 1/3, 1/6])

        return cls(cv, bv, rkm, fun, x0)

    @classmethod
    def three_eight_rule(cls, fun, x0):
        rkm = np.array([[1/3, 0, 0],
                        [-1/3, 1, 0], 
                        [1, -1, 1]]
                       )

        cv = np.array([0, 1/3, 2/3, 1])
        bv = np.array([1/8, 3/8, 3/8, 1/8])

        return cls(cv, bv, rkm, fun, x0)

    def __next__(self):
        h = self.step_size
        s = self.nsteps

        kv = [] 

        t = self.x[0]
        y = self.x[1]

        for n in range(1, s+1):
            kv.append(
                    self.fun(
                        [t + self.cv[n-1]*h, y + 
                         h*self.rkm[n-2,0:n-1]@np.array(kv)]
                        )
                    )

        self.x = [t + h, y + h*self.bv@kv]

        return self.x

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def fun(x):
        return -x[1]

    x0 = [0, 1]
    rkint = RKIntegrator(cv, bv, rkm, fun, x0)

    X = []
    while rkint.x[0] < 100:
        X.append(next(rkint))

    X = np.array(X)

    plt.plot(X[:,0], X[:,1], '.')
    plt.plot(X[:,0], np.exp(-X[:,0]))
    plt.show()

