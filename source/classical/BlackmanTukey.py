import cmath
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal as signal
import csv
from ..utils.Autocorrelation import Autocorrelation

sns.set()

class BlackmanTukey():

    def __init__(self):
        self.f = None
        self.default_f = np.linspace(0, 0.5, 500)
        self.x = None
        self.P = None
        self.w = None
        self.r_xx = Autocorrelation()

    def estimate(self, x, f=None, M=10):
        '''
            Estimates P_xx for given signal sequence x.

            Args:
                x (numpy array of doubles): Signal
                f (numpy array of doubles in range [0, 0.5]): Frequence
                M (integer): Window is defined as triangle in interval [-M, M]
        '''
        # Init values.
        if f is None:
            self.f = self.default_f
        else:
            self.f = f
        self.x = x
        self.M = M
        self.N = len(self.x)
        self.P = np.zeros(len(self.f))

        # Init window and estimate autocerr. function.
        w = TriangWindow(M)
        self.r_xx.estimate(x)

        # Calculate P.
        for i in range(len(self.f)):
            sum = 0
            for n in np.arange(-self.N + 1, self.N):
                sum += w[n] * self.r_xx[n] * cmath.exp(-1j * 2*cmath.pi * self.f[i] * n)
            self.P[i] = sum.real

    def plot(self):
        '''
            Plots estimated P.
        '''
        # Check if anything is estimated.
        if self.f is None or self.P is None:
            return

        plt.figure()
        plt.semilogy(self.f, self.P)
        plt.title('Blackman-Tukey method estimation')
        plt.xlabel('f [Hz]')
        plt.ylabel('P')
        plt.show()

    def compare(self, x, M=10):
        '''
            Compares with Blackman-Tukey method from scipy.signal by ploting them both.

            Args:
                x (numpy array of doubles): Signal
                M (integer): Window is defined as triangle in interval [-M, M]
        ''' 
        wd = signal.get_window('triang', 2*M + 1)
        f_per, P_per = signal.welch(x, scaling='spectrum', nperseg=len(x), noverlap=0, window=wd)            
        self.estimate(x, f_per, M)

        # Plot them together.
        plt.figure()
        plt.semilogy(self.f, self.P, 'b', label='Blackman-Tukey method')
        plt.semilogy(f_per, P_per, 'r--', label='scipy.signal')
        plt.legend()
        plt.title('Blackman-Tukey method comparation')
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
        if key == 'M':
            return self.M
        return None

class TriangWindow():

    def __init__(self, M):
        if M < 0:
            return
        self.M = M

        self.window = np.zeros(self.M + 1)
        for n in range(M + 1):
            self.window[n] = -n / (self.M + 1) + 1

    def plot(self):
        '''
            Plots triangle window.
        '''
        window_plot = self._double_side_window()
        window_plot = np.append([0], window_plot)
        window_plot = np.append(window_plot, [0])

        plt.figure()
        plt.plot(np.arange(-len(window_plot)//2 + 1, len(window_plot)//2 + 1), window_plot)
        plt.show()

    def _double_side_window(self):
        window_rev = self.window[::-1]
        return np.append(window_rev[:-1], self.window)

    def __getitem__(self, key):
        '''
            Returns window[key].
        '''
        if self.window is None:
            return None

        abs_key = abs(key)
        if abs_key > self.M:
            return 0
        return self.window[abs_key]