import cmath
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal as signal
from ..utils.Autocorrelation import Autocorrelation

sns.set()

class AutocorrelationMethod():

    def __init__(self):
        self.f = None
        self.default_f = np.linspace(0, 0.5, 500)
        self.x = None
        self.p = None
        self.a = None
        self.var_u = None
        self.P = None

        self.r_xx = Autocorrelation()

    def estimate(self, x, f=None, p=2):
        '''
            Estimates P_xx for given signal sequence x.

            Args:
                x (numpy array of doubles): Signal
                f (numpy array of doubles): Frequence
                p (integer): Polinomial order in A(f)
        '''
        # Init values
        self.x = x
        if f is None:
            self.f = self.default_f
        else:
            self.f = f
        self.p = p
        self.P = np.zeros(len(self.f))

        # Estimate autoccorelation
        self.r_xx.estimate(x)

        # Compose matrix Rxx
        Rxx = np.zeros([p, p])
        for i in range(p):
            for j in range(p):
                Rxx[i][j] = self.r_xx[i - j]

        # Compose vector rxx
        rxx_vec = np.zeros([p, 1])
        for i in range(p):
            rxx_vec[i] = self.r_xx[i + 1]

        # Calculate a and append [1] for a[0]
        self.a = np.matmul(-np.linalg.inv(Rxx),
                           rxx_vec)
        self.a = np.append([[1]], self.a, axis=0)

        # Calculate var_u
        self.var_u = 0
        for i in range(p + 1):
            self.var_u += self.a[i] * self.r_xx[-i]

        # Calculate P
        for fi in range(len(self.f)):
            A = 0
            for i in range(p + 1):
                A += self.a[i] * cmath.exp(-1j * 2*cmath.pi * self.f[fi] * i)
            self.P[fi] = self.var_u / pow(abs(A), 2)

    def plot(self):
        '''
            Plots estimated P.
        '''
        # Check if anything is estimated.
        if self.f is None or self.P is None:
            return

        plt.figure()
        plt.semilogy(self.f, self.P)
        plt.title('Autocorrelation method estimation')
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