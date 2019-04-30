import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv

sns.set()

class Autocorrelation():

    def __init__(self):
        self.x = None
        self.N = None
        self.r_xx = None

    def estimate(self, x):
        '''
            Estimates r_xx for given sequence x.

            Args:
                x (numpy array of doubles): Signal
        '''
        # Init values.
        self.x = x
        self.N = len(self.x)
        self.r_xx = np.zeros(self.N)

        # Calculate r_xx.
        for n in range(self.N):
            sum = 0
            for m in range(self.N - 1 - n):
                sum += np.conjugate(self.x[m]) * self.x[m + n]
            self.r_xx[n] = sum / self.N

    def plot(self):
        '''
            Plots estimated r_xx.
        '''
        # Check if anything is estimated.
        if self.r_xx is None:
            return
        r_xx_plot = self._double_side_r_xx()

        plt.figure()
        plt.plot(np.arange(-len(r_xx_plot)//2 + 1, len(r_xx_plot)//2 + 1), r_xx_plot)
        plt.title('Autocorrelation function estimation')
        plt.xlabel('n')
        plt.ylabel('r_xx')
        plt.show()

    def compare(self, x=None):
        '''
            Compares with numpy functon 'correlate'.

            Args:
                x (numpy array of doubles): Signal
        ''' 
        r_xx_np = np.correlate(x, x, mode='full')           
        self.estimate(x)
        r_xx_plot = self._double_side_r_xx()

        # Plot them together.
        plt.figure()
        plt.plot(np.arange(-len(r_xx_plot)//2 + 1, len(r_xx_plot)//2 + 1), r_xx_plot,
            'b', label='autocorrelation')
        plt.plot(np.arange(-len(r_xx_np)//2 + 1, len(r_xx_np)//2 + 1), r_xx_np,
            'r--', label='numpy')
        plt.legend()
        plt.title('Autocorrelation function comparation')
        plt.xlabel('n')
        plt.ylabel('r_xx')
        plt.show()

    def _double_side_r_xx(self):
        r_xx_rev = self.r_xx[::-1]
        return np.append(r_xx_rev[:-1], self.r_xx)

    def __getitem__(self, key):
        '''
            Returns r_xx[key].
        '''
        if self.r_xx is None:
            return None

        if abs(key) >= self.N:
            return 0
        if key >= 0:
            return self.r_xx[key]
        else:
            return np.conjugate(self.r_xx[-key])