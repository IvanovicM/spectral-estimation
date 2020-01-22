import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

class ModelOrderSelector():

    def __init__(self):
        self.method_values = None

    def apply(self, method, x, max_order, estimator):
        '''
            Applies method for model order selection by calculationg function
            value for each order in given interval.

            Args:
                method (string): Method. It can be 'FPE', 'AIC' and 'CAT'.
                x (array of doubles): Signal.
                max_order (integer): Maximum order for the method.
                estimator (parametric estimator): One of parametric estimators.
        '''
        if method is None or estimator is None:
            return
        if method == 'FPE' or method == 'AIC' or method == 'CAT':
            self.method = method
            self.max_order = max_order
            self.rho = np.zeros(max_order + 1)
            self.x = x
            self.N = len(x)
        else:
            return

        # Apply estimator.
        for k in np.arange(1, self.max_order + 1):
            print('Applying for k={}'.format(k))
            estimator.estimate(self.x, p=k)
            self.rho[k] = estimator['var_u']

        # Apply method.
        self.method_values = np.zeros(self.max_order + 1)
        for k in np.arange(1, self.max_order + 1):
            self.method_values[k] = self._apply_for_given_k(k, self.rho[k])

    def plot(self):
        '''
            Plots method values for order in interval [1, max_order].
        '''
        # Check if anything is calculated.
        if self.method is None or self.method_values is None or self.max_order is None:
            return

        # Plot all.
        plt.figure()
        plt.plot(np.arange(1, self.max_order + 1), self.method_values[1:])

        # Mark the optimal value and order.
        min_indx = np.argmin(self.method_values[1:])
        k_opt = min_indx + 1
        plt.plot(k_opt, self.method_values[k_opt], 'ro',
                 label='Min value for k_opt={}'.format(k_opt))

        plt.title('Criterion in model order selection')
        plt.xlabel('k')
        plt.ylabel('{}[k]'.format(self.method))
        plt.legend()
        plt.show()

    def _apply_for_given_k(self, k, rho_k):
        if self.method == 'FPE':
            return (self.N + k) / (self.N - k) * rho_k
        elif self.method == 'AIC':
            return self.N * math.log(rho_k) + 2 * k
        elif self.method == 'CAT':
            rho_tilda_k = self.N / (self.N - k) * rho_k

            if k == 1:
                self.rho_tilda_sum = 1 / rho_tilda_k
            else:
                self.rho_tilda_sum += 1 / rho_tilda_k

            return self.rho_tilda_sum / self.N - 1 / rho_tilda_k
        else:
            return None

    def __getitem__(self, key):
        if key == 'method':
            return self.method
        if key == 'values':
            return self.method_values
        return None