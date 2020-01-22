import numpy as np
from scipy import signal
from ..parametric.AutocorrelationMethod import AutocorrelationMethod
from ..parametric.CovarianceMethod import CovarianceMethod
from ..parametric.ModifiedCovarianceMethod import ModifiedCovarianceMethod
from ..parametric.Burg import Burg

def autocorrelation_method_test():
    t = np.arange(1000)
    x = np.sin(1*t + 2.8) + np.sin(2*t + 3.4)
    autocorr_method = AutocorrelationMethod()

    # Estimation
    autocorr_method.estimate(x, p=4)
    autocorr_method.plot()

    print('AutocorrelationMethod, var_u: ', autocorr_method['var_u'])

def covariance_method_test():
    t = np.arange(1000)
    x = np.sin(1*t + 2.8) + np.sin(2*t + 3.4)
    u = np.random.normal(size=1000)
    cov_method = CovarianceMethod()

    # Estimation
    cov_method.estimate(np.add(x, u), p=4)
    cov_method.plot()

    print('CovarianceMethod, var_u: ', cov_method['var_u'])

def modified_covariance_method_test():
    t = np.arange(1000)
    x = np.sin(1*t + 2.8) + np.sin(2*t + 3.4)
    u = np.random.normal(size=1000)
    mod_cov_method = ModifiedCovarianceMethod()

    # Estimation
    mod_cov_method.estimate(np.add(x, u), p=4)
    mod_cov_method.plot()

    print('ModifiedCovarianceMethod, var_u: ', mod_cov_method['var_u'])

def burg_method_test():
    t = np.arange(256)
    x = np.sin(1*t + 2.8) + np.sin(2*t + 3.4)
    u = np.random.normal(size=256)
    burg = Burg()

    # Estimation
    burg.estimate(np.add(x, u), p=10)
    burg.plot()

if __name__ == "__main__":
    autocorrelation_method_test()
    covariance_method_test()
    modified_covariance_method_test()
    burg_method_test()