import numpy as np
from scipy import signal
from parametric.CovarianceMethod import CovarianceMethod 
from utils.MeanAndVar import MeanAndVar
from utils.ModelOrderSelector import ModelOrderSelector

def model_order_selector_test():
    N = 256
    u = np.random.normal(size=N)

    b = [2,3]
    a = [1,0.2]
    x = signal.lfilter(b, a, u)

    # Apply order selection on Covariance method
    selector = ModelOrderSelector()
    cov = CovarianceMethod()
    selector.apply('CAT', x, 30, cov)
    selector.plot()

def mean_var_test():
    x = np.random.normal(size=[100, 150])
    mv = MeanAndVar()
    mv.estimate(x)
    mv.plot(np.arange(-200, -50))

if __name__ == "__main__":
    mean_var_test()
    model_order_selector_test()