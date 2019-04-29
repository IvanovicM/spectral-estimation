import numpy as np
from parametric.AutocorrelationMethod import AutocorrelationMethod

def autocorrelation_method_test():
    x = np.random.normal(size=100)
    autocorr_method = AutocorrelationMethod()

    # Estimation
    autocorr_method.estimate(x)
    autocorr_method.plot()

if __name__ == "__main__":
    autocorrelation_method_test()