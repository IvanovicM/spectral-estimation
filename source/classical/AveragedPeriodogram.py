import cmath
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal as signal
import csv
from .Periodogram import Periodogram

sns.set()

class AveragedPeriodogram():

    def __init__(self):
        self.f = None
        self.default_f = np.linspace(0, 0.5, 500)
        self.x = None
        self.P = None
        self.L = None
        self.K = None
        self.per = Periodogram()

    def estimate(self, x, f=None, L=None):
        '''
            Estimates P_xx for given signal sequence x by dividing it into segment.
            If the length of a segment is not given, it is by default equals to the length of x.

            Args:
                x (numpy array of doubles): Signal
                f (numpy array of doubles in range [0, 0.5]): Frequence
                L (integer): The length of each segment
        '''
        # Init values.
        if f is None:
            self.f = self.default_f
        else:
            self.f = f
        self.x = x
        self.N = len(self.x)
        self.P = np.zeros(len(self.f))
        if L is None or L > self.N:
            self.L = self.N 
        else:
            self.L = L
        self.K = self.N // self.L

        # Calculate P.
        for num_segment in range(self.K):
            x_segment = x[num_segment*self.L : (num_segment+1)*self.L]
            self.per.estimate(x_segment, self.f)
            self.P = np.add(self.P, self.per['P'])
        self.P = self.P / self.K

    def plot(self):
        '''
            Plots estimated P.
        '''
        # Check if anything is estimated.
        if self.f is None or self.P is None:
            return

        plt.figure()
        plt.semilogy(self.f, self.P)
        plt.title('Averaged periodogram estimation')
        plt.xlabel('f [Hz]')
        plt.ylabel('P')
        plt.show()

    def compare(self, x, L=None):
        '''
            Compares with periodogram from scipy.signal by ploting them both.

            Args:
                x (numpy array of doubles): Signal
                L (integer): The length of each segment
        ''' 
        f_per, P_per = signal.welch(x, scaling='spectrum', nperseg=L, noverlap=0, window='boxcar')           
        self.estimate(x, f_per, L)

        # Plot them together.
        plt.figure()
        plt.semilogy(self.f, self.P, 'b', label='averaged periodogram')
        plt.semilogy(f_per, P_per, 'r--', label='scipy.signal')
        plt.legend()
        plt.title('Averaged periodogram comparation')
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
        if key == 'L':
            return self.L
        return None