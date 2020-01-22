import cmath
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal as signal
from ..utils.Autocorrelation import Autocorrelation

sns.set()

class Burg():

    def __init__(self):
        self.f = None
        self.default_f = np.linspace(0, 0.5, 500)
        self.x = None

        self.P = None
        self.rho = None
        self.e_b = None
        self.e_f = None
        self.refl_coeff = None
        self.a = None
        self.p = None

        self.r_xx = Autocorrelation()

    def estimate(self, x, f=None, p=5):
        '''
            Estimates P_xx for given signal sequence x.

            Args:
                x (numpy array of doubles): Signal
                f (numpy array of doubles): Frequence
                p (integer): Max polinomial order in A(f)
        '''
        # Init values
        self.x = x
        self.N = len(x)
        if f is None:
            self.f = self.default_f
        else:
            self.f = f
        self.p = p
        self.P = np.zeros([p + 1, len(self.f)])

        # Init all vectors and matrices.
        self.e_b = np.zeros([self.p + 1, self.N])
        self.e_f = np.zeros([self.p + 1, self.N])
        self.a = np.zeros([self.p + 1, self.p + 1])
        self.rho = np.zeros(self.p + 1)
        self.refl_coeff = np.zeros(self.p + 1)

        # Calculate variance and estimated error for order 0.
        self.r_xx.estimate(x)
        self.rho[0] = self.r_xx[0]
        
        for n in range(self.N):
            self.e_b[0][n] = self.x[n]
            self.e_f[0][n] = self.x[n]

        # Calcualte coefficients for every order in [1, p].
        for k in np.arange(1, self.p + 1):
            # Calculate reflection coefficient.
            num = 0
            den = 0
            for n in np.arange(k, self.N):
                num += (self.e_f[k - 1][n] *
                        np.conjugate(self.e_b[k - 1][n - 1]))
                den += (pow(abs(self.e_f[k - 1][n]), 2) +
                        pow(abs(self.e_b[k - 1][n - 1]), 2))
            self.refl_coeff[k] = -2 * num / den

            # Calculate prediction variance.
            self.rho[k] = (1 - pow(abs(self.refl_coeff[k]), 2) ) * self.rho[k - 1]

            # Calculate polinomial coefficients for this order.
            for i in np.arange(1, k):
                self.a[k][i] = (self.a[k - 1][i] +
                                self.refl_coeff[k] * np.conjugate(self.a[k - 1][k - i]))
            self.a[k][k] = self.refl_coeff[k]
            self.a[k][0] = 1

            # Calculate estimated errors.
            for n in range(self.N):
                self.e_b[k][n] = (self.e_b[k - 1][n - 1] +
                                  np.conjugate(self.refl_coeff[k]) * self.e_f[k - 1][n])
                self.e_f[k][n] = (self.e_f[k - 1][n] +
                                  self.refl_coeff[k] * self.e_b[k - 1][n - 1])

        # Estimate P for every order in [1, p]
        for k in np.arange(1, self.p + 1):
            for fi in range(len(self.f)):
                A = 0
                for i in range(k + 1):
                    A += self.a[k][i] * cmath.exp(-1j * 2*cmath.pi * self.f[fi] * i)
                self.P[k][fi] = self.rho[k] / pow(abs(A), 2)

    def plot(self):
        '''
            Plots estimated P for every order.
        '''
        # Check if anything is estimated.
        if self.f is None or self.P is None:
            return

        plt.figure()
        for k in np.arange(1, self.p + 1):
            plt.semilogy(self.f, self.P[k], label='p = {}'.format(k))
        plt.title('Burg method')
        plt.legend()
        plt.xlabel('f [Hz]')
        plt.ylabel('P')
        plt.show()

    def __getitem__(self, key):
        '''
            Returns the value for given key, or None if the key is not allowed.
        '''
        if key == 'f':
            return self.f
        if key == 'P':
            return self.P
        if key == 'x':
            return self.x
        if key == 'rho':
            return self.rho
        if key == 'refl_coeff':
            return self.refl_coeff
        if key == 'a':
            return self.a
        return None