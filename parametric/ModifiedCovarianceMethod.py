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

    def estimate(self, x, f=None):
        '''
            Estimates P_xx for given signal sequence x.

            Args:
                x (numpy array of doubles): Signal
        '''
        self.x = x

    def __getitem__(self, key):
        return 0