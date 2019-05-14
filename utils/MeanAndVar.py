import numpy as np
import math

class MeanAndVar:

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
            
        self.Nr = np.shape(x)[0]
        self.N = np.shape(x)[1]
        if self.Nr == 0 or self.N == 0:
            return

        # Estimate mean.
        self.mean = np.zeros(self.N)
        for nr in range(self.Nr):
            self.mean = np.add(self.mean, x[nr][:])
        self.mean /= self.Nr

        # Estimate var.
        self.var = np.zeros(self.N)
        for nr in range(self.Nr):
            for n in range(self.N):
                self.var[n] += pow(x[nr][n] - self.mean[n], 2)
        self.var /= (self.Nr - 1)

    def __getitem__(self, key):
        if key == 'mean':
            return self.mean
        if key == 'var':
            return self.var
        if key == 'stddev':
            return math.sqrt(self.var)
        return None