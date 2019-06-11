import cmath
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal as signal

sns.set()

class ModifiedCovarianceMethod():

    def __init__(self):
        self.f = None
        self.default_f = np.linspace(0, 0.5, 500)
        self.x = None
        self.N = None
        self.p = None
        self.a = None
        self.var_u = None
        self.P = None
        self.c_xx = None

    def estimate(self, x, f=None, p=2):
        '''
            Estimates P_xx for given signal sequence x.

            Args:
                x (numpy array of doubles): Signal
                f (numpy array of doubles): Frequence
                p (integer): Polinomial order in A(f)
        '''
        # Initi values
        self.x = x
        self.N = len(x)
        if f is None:
            self.f = self.default_f
        else:
            self.f = f
        self.p = p
        self.P = np.zeros(len(self.f))

        # Estimate c_xx
        self._estimate_c_xx()

        # Compose matrix C
        C = np.zeros([p, p])
        for j in range(p):
            for k in range(p):
                C[j][k] = self.c_xx[j + 1][k + 1]

        # Compose vector cxx
        cxx_vec = np.zeros([p, 1])
        for j in range(p):
            cxx_vec[j] = self.c_xx[j + 1][0]

        # Calculate a and append [1] for a[0]
        self.a = np.matmul(-np.linalg.inv(C),
                           cxx_vec)
        self.a = np.append([[1]], self.a, axis=0)

        # Calculate var_u
        self.var_u = 0
        for k in range(p + 1):
            self.var_u += self.a[k] * self.c_xx[0][k]

        # Calculate P
        for fi in range(len(self.f)):
            A = 0
            for i in range(p + 1):
                A += self.a[i] * cmath.exp(-1j * 2*cmath.pi * self.f[fi] * i)
            self.P[fi] = self.var_u / pow(abs(A), 2)

    def _estimate_c_xx(self):
        self.c_xx = np.zeros([self.p + 1, self.p + 1])

        # Calculate c_xx.
        for j in range(self.p + 1):
            for k in range(self.p + 1):
                sum = 0

                # Forward
                for n in np.arange(self.p, self.N):
                    sum += np.conjugate(self.x[n - j]) * self.x[n - k]
                # Backward
                for n in np.arange(0, self.N - self.p):
                    sum += self.x[n + j] * np.conjugate(self.x[n + k])
                    
                self.c_xx[j][k] = sum / (self.N - self.p)

    def plot(self):
        '''
            Plots estimated P.
        '''
        # Check if anything is estimated.
        if self.f is None or self.P is None:
            return

        plt.figure()
        plt.semilogy(self.f, self.P)
        plt.title('Modified covariance method estimation')
        plt.xlabel('f [Hz]')
        plt.ylabel('P')
        plt.show()

    def __getitem__(self, key):
        if key == 'f':
            return self.f
        if key == 'P':
            return self.P
        if key == 'x':
            return self.x
        if key == 'a':
            return self.a
        if key == 'var_u':
            return self.var_u
        if key == 'p':
            return self.p
        return None