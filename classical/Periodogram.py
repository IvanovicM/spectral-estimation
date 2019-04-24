import cmath
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal
import csv

sns.set()

class Periodogram():

    def __init__(self):
        self.f = None
        self.default_f = np.linspace(0, 0.5, 500)
        self.x = None
        self.P = None

    def estimate(self, x, f=None):
        '''
            Estimates P_xx for given signal sequence x.

            Args:
                x (numpy array of doubles): Signal
                f (numpy array of doubles in range [0, 0.5]): Frequence
        '''
        # Init values.
        if f is None:
            self.f = self.default_f
        else:
            self.f = f
        self.x = x
        self.N = len(self.x)
        self.P = np.zeros(len(self.f))

        # Calculate P.
        for i in range(len(self.f)):
            sum = 0
            for n in range(self.N):
                sum += self.x[n] * cmath.exp(-1j * 2*cmath.pi * self.f[i] * n)
            self.P[i] = pow(abs(sum), 2) / self.N

    def plot(self):
        '''
            Plots estimated P.
        '''
        # Check if anything is estimated.
        if self.f is None or self.P is None:
            return

        plt.figure()
        plt.semilogy(self.f, self.P)
        plt.title('Periodogram estimation')
        plt.xlabel('f [Hz]')
        plt.ylabel('P')
        plt.show()

    def compare(self, x=None):
        '''
            Compares with periodogram from scipy.signal by ploting them both.

            Args:
                x (numpy array of doubles): Signal
        ''' 
        f_per, P_per = scipy.signal.periodogram(x, scaling='spectrum')             
        self.estimate(x, f_per)

        # Plot them together.
        plt.figure()
        plt.semilogy(self.f, self.P, 'b', label='periodogram')
        plt.semilogy(f_per, P_per, 'r--', label='scipy.signal')
        plt.legend()
        plt.title('Periodogram comparation')
        plt.xlabel('f [Hz]')
        plt.ylabel('P')
        plt.ylim(bottom=1e-5)
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
        return None