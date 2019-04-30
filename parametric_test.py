import numpy as np
from scipy import signal
from parametric.Autocorrelation import Autocorrelation
from parametric.AutocorrelationMethod import AutocorrelationMethod

def autocorrelation_method_test():
    u = np.random.normal(size=256)
    b = [1]
    a = [1, -2.760, 3.809, -2.654, 0.924]
    x = signal.filtfilt(b, a, u)
    autocorr_method = AutocorrelationMethod()

    # Estimation
    autocorr_method.estimate(x, p=4)
    autocorr_method.plot()

def autocorrelation_test():
    x = np.random.normal(size=1000)
    r_xx = Autocorrelation()

    # Estimation
    r_xx.estimate(x)
    r_xx.plot()

    # Comparation
    r_xx.compare(x)

if __name__ == "__main__":
    #autocorrelation_test()
    autocorrelation_method_test()