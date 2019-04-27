import numpy as np
from classical.Periodogram import Periodogram
from classical.AveragedPeriodogram import AveragedPeriodogram
from classical.Autocorrelation import Autocorrelation

def periodogram_test():
    x = np.random.normal(size=100)
    per = Periodogram()

    # Estimation
    per.estimate(x)
    per.plot()

    # Comparation
    per.compare(x)

def averaged_periodogram_test():
    x = np.random.normal(size=1000)
    per = AveragedPeriodogram()

    # Estimation
    per.estimate(x, L=100)
    per.plot()

    # Comparation
    per.compare(x, 100)

def autocorrelation_test():
    x = np.random.normal(size=1000)
    r_xx = Autocorrelation()

    # Estimation
    r_xx.estimate(x)
    r_xx.plot()

    # Comparation
    r_xx.compare(x)

if __name__ == "__main__":
    #periodogram_test()
    #averaged_periodogram_test()
    autocorrelation_test()