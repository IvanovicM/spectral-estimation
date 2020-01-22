import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

class MeanAndVar():

    def __init__(self):
        self.mean = None
        self.var = None

    def estimate(self, x):
        '''
            Estimates mean and var for given realisations of some process.

            Args:
                x (numpy matrix of doubles): Given reaslisations of a proces.
                    The first dimension is number of reaslisations.
                    Second dimension is number of samples per realisaton.
        '''
        if x is None:
            return
            
        x_shape = np.shape(x)
        self.Nr = x_shape[0]
        if np.shape(x_shape)[0] == 1:
            self.N = 1
        else:
            self.N = x_shape[1]
        if self.Nr == 0 or self.N == 0:
            return

        # Estimate mean.
        self.mean = np.zeros(self.N)
        for nr in range(self.Nr):
            if self.N == 1:
                curr_x = x[nr]
            else:
                curr_x = x[nr][:]
            self.mean = np.add(self.mean, curr_x)
        self.mean /= self.Nr

        # Estimate var.
        self.var = np.zeros(self.N)
        for nr in range(self.Nr):
            for n in range(self.N):
                if self.N == 1:
                    curr_x = x[nr]
                else:
                    curr_x = x[nr][n]
                self.var[n] += pow(curr_x - self.mean[n], 2)
        self.var /= (self.Nr - 1)

    def plot(self, x_label=None):
        '''
            Plots estimated mean with sigma around it.

            Args:
                x_label (numpy arayy of doubes): Optional label for x-axis.
        '''
        if self.mean is None or self.var is None:
            return

        if x_label is None or len(x_label) != len(self.mean):
            x_label = np.arange(0, self.N)

        # Plot mean +- sigma.
        ax1 = plt.subplot(212)
        ax1.plot(x_label, self.mean, 'k')
        ax1.plot(x_label, np.add(self.mean, np.sqrt(self.var)), 'r--')
        ax1.plot(x_label, np.add(self.mean, -np.sqrt(self.var)), 'r--')
        plt.title('mean +- sigma')

        # Plot mean
        ax2 = plt.subplot(221)
        ax2.plot(x_label, self.mean, 'b')
        ax2.set_title('mean')
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        # Plot variancce.
        ax3 = plt.subplot(222)
        ax3.plot(x_label, self.var, 'g')
        ax3.set_title('variance')
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        
        plt.show()

    def __getitem__(self, key):
        if key == 'mean':
            return self.mean
        if key == 'var':
            return self.var
        if key == 'stddev':
            return np.sqrt(self.var)
        return None