import numpy as np
from scipy import signal
from parametric.Autocorrelation import Autocorrelation
from parametric.AutocorrelationMethod import AutocorrelationMethod
from parametric.CovarianceMethod import CovarianceMethod
from parametric.ModifiedCovarianceMethod import ModifiedCovarianceMethod
from parametric.Burg import Burg
from utils.MeanAndVar import MeanAndVar
from utils.ModelOrderSelector import ModelOrderSelector

def autocorrelation_method_test():
    t = np.arange(1000)
    x = np.sin(1*t + 2.8) + np.sin(2*t + 3.4)
    autocorr_method = AutocorrelationMethod()

    # Estimation
    autocorr_method.estimate(x, p=4)
    autocorr_method.plot()

    print('AutocorrelationMethod, var_u: ', autocorr_method['var_u'])

def autocorrelation_test():
    x = np.random.normal(size=1000)
    r_xx = Autocorrelation()

    # Estimation
    r_xx.estimate(x)
    r_xx.plot()

    # Comparation
    r_xx.compare(x)

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

def model_order_selector_test():
    N = 1000
    t = np.arange(N)
    x = np.sin(1*t + 2.8) + np.sin(2*t + 3.4)
    u = np.random.normal(size=N)

    mv = MeanAndVar()
    selector = ModelOrderSelector()
    autocorr = ModifiedCovarianceMethod()

    # Estimation + variance for various p
    max_p = 2
    rho = np.zeros(max_p + 1)
    for p in np.arange(1, max_p + 1):
        autocorr.estimate(np.add(x, u), p=p)
        mv.estimate(np.transpose(autocorr['P']))
        rho[p] = mv['var']

    # Apply order selection
    selector.apply('FPE', N, max_p, rho)
    selector.plot()

def mean_var_test():
    x = np.random.normal(size=[100, 150])
    mv = MeanAndVar()
    mv.estimate(x)
    mv.plot(np.arange(-200, -50))

if __name__ == "__main__":
    #autocorrelation_test()
    #autocorrelation_method_test()
    #covariance_method_test()
    #modified_covariance_method_test()
    burg_method_test()
    #mean_var_test()
    #model_order_selector_test()