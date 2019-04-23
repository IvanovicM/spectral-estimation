import numpy as np
from classical.Periodogram import Periodogram

def periodogram_test():
    x = np.random.normal(size=100)
    per = Periodogram()

    # Estimation
    per.estimate(x)
    per.plot()

    # Comparation
    per.compare(x)

if __name__ == "__main__":
    periodogram_test()