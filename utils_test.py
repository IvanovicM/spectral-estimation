import numpy as np
from scipy import signal
from parametric.CovarianceMethod import CovarianceMethod 
from utils.MeanAndVar import MeanAndVar
from utils.ModelOrderSelector import ModelOrderSelector

def model_order_selector_test():
    N = 256
    t = np.arange(N)
    u = np.random.normal(size=N)

    b = [2,3]
    a = [1,0.2]
    x = signal.lfilter(b, a, u)

    selector = ModelOrderSelector()
    cov = CovarianceMethod()

    # Estimation + variance for various p
    max_p = 20
    rho = np.zeros(max_p + 1)
    for p in np.arange(1, max_p + 1):
        cov.estimate(x, p=p)
        rho[p] = cov['var_u']

    # Apply order selection
    selector.apply('CAT', N, max_p, rho)
    selector.plot()

def mean_var_test():
    x = np.random.normal(size=[100, 150])
    mv = MeanAndVar()
    mv.estimate(x)
    mv.plot(np.arange(-200, -50))

if __name__ == "__main__":
    mean_var_test()
    model_order_selector_test()