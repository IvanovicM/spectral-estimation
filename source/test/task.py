import numpy as np 
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from random import randint
from scipy import signal

from ..classical.Periodogram import Periodogram
from ..classical.AveragedPeriodogram import AveragedPeriodogram
from ..classical.BlackmanTukey import BlackmanTukey
from ..parametric.AutocorrelationMethod import AutocorrelationMethod
from ..parametric.Burg import Burg
from ..parametric.CovarianceMethod import CovarianceMethod
from ..parametric.ModifiedCovarianceMethod import ModifiedCovarianceMethod
from ..utils.Autocorrelation import Autocorrelation
from ..utils.MeanAndVar import MeanAndVar
from ..utils.ModelOrderSelector import ModelOrderSelector

sns.set()

def read_data(filename, delimiter, file_size):
    # Create reader
    data_file = open(filename)
    data_csv = csv.reader(data_file, delimiter=delimiter)

    # Init data
    x = np.zeros(file_size)
    file_column_count = file_size[1]

    # Read data
    row = 0
    for data_row in data_csv:
        for column in range(file_column_count):
            x[row][column] = data_row[column]
        row += 1
        
    return x

def plot_realisations(x, num=1):
    plt.figure()
    for i in range(num):
        plt.plot(x[i][:], label=i)
    plt.legend()
    plt.title('Data')
    plt.xlabel('t [s]')
    plt.show()

def apply_classical_methods(x):
    # Periodogram
    rand_index = randint(0, np.shape(x)[0] - 1)
    per = Periodogram()
    per.estimate(x[rand_index][:])
    per.plot()
    per.compare(x[rand_index][:])

    # Averaged periodogram
    rand_index = randint(0, np.shape(x)[0] - 1)
    avg_per = AveragedPeriodogram()
    avg_per.estimate(x[rand_index][:], L=50)
    avg_per.plot()
    avg_per.compare(x[rand_index][:], L=50)

    # Blackman-Tukey
    rand_index = randint(0, np.shape(x)[0] - 1)
    bm = BlackmanTukey()
    bm.estimate(x[rand_index][:], M=50)
    bm.plot()

def apply_parametric_methods(x):
    # Autocorrelaton method
    rand_index = randint(0, np.shape(x)[0] - 1)
    autocorr = AutocorrelationMethod()
    autocorr.estimate(x[rand_index][:], p=15)
    autocorr.plot()

    # Covariance method
    rand_index = randint(0, np.shape(x)[0] - 1)
    cov = CovarianceMethod()
    cov.estimate(x[rand_index][:], p=15)
    cov.plot()
        
    # Modified covariance method
    rand_index = randint(0, np.shape(x)[0] - 1)
    mod_cov = ModifiedCovarianceMethod()
    mod_cov.estimate(x[rand_index][:], p=15)
    mod_cov.plot()

    # Burg method
    rand_index = randint(0, np.shape(x)[0] - 1)
    burg = Burg()
    burg.estimate(x[rand_index][:], p=40)
    burg.plot()

def plot_all_with_variance(estimator, x, title_0, title_1, p=None, M=None):
    Nr = np.shape(x)[0]
    N = np.shape(x)[1]
    mv = MeanAndVar()
    f_len = 100
    f = np.linspace(0, 0.5, f_len)
    estimated_P = np.zeros(shape=[Nr, f_len])

    _, axarr = plt.subplots(1, 2)
    for nr in range(Nr):
        # Plot estimated P.
        print('Estimating:', nr)
        if p is None:
            if M is None:
                estimator.estimate(x[nr][:], f=f)
            else:
                estimator.estimate(x[nr][:], f=f, M=M)
        else:
            estimator.estimate(x[nr][:], f=f, p=p)
        
        axarr[0].semilogy(f, estimator['P'])
        estimated_P[nr][:] = estimator['P']

    # Plot variance.
    mv.estimate(estimated_P)
    axarr[1].plot(f, mv['var'])

    # Label subplots & show
    axarr[0].set_title(title_0)
    axarr[0].set(xlabel='f [Hz]', ylabel='P [dB]')
    axarr[1].set_title(title_1)
    axarr[1].set(xlabel='f [Hz]', ylabel='var')
    plt.show()

def window_closing_on_blackman_tukey(x):
    rand_index = randint(0, np.shape(x)[0] - 1)
    xr = x[rand_index][:]

    # Window closing and ploting
    bm = BlackmanTukey()
    plt.figure()
    for M in [10, 20, 40, 80, 160]:
        bm.estimate(xr, M=M)
        plt.semilogy(bm['f'], bm['P'], label=M)
    plt.title('Blackman-Tukey window closing')
    plt.legend()
    plt.xlabel('f [Hz]')
    plt.ylabel('P')
    plt.show()

def apply_and_plot_all(x):
    # Periodogram
    plot_all_with_variance(Periodogram(), x,
                           'Periodogram on all realisations',
                           'Variance for periodogram')

    # CovarianceMethod
    plot_all_with_variance(CovarianceMethod(), x,
                           'Covariance method on all realisations',
                           'Variance for covariance method',
                           p=5)

    # Blackman-Tukey
    plot_all_with_variance(BlackmanTukey(), x,
                           'BlackMan-Tukey method on all realisations',
                           'Variance for BlackMan-Tukey method',
                           M=10)

def apply_and_plot_with_order(x, p):  
    cov = CovarianceMethod()  
    for p_i in p:
        plot_all_with_variance(cov, x,
                               'Covariance method for p = {}'.format(p_i),
                               'Variance for p = {}'.format(p_i),
                               p=p_i)

def show_variance_for_covariance_method(x, p):
    Nr = np.shape(x)[0]
    N = np.shape(x)[1]
    cov = CovarianceMethod()
    mv = MeanAndVar()

    _, axarr = plt.subplots(1, 2)
    i_ax = 0
    f_len = 100
    f = np.linspace(0, 0.5, f_len)

    for curr_N in [N, N // 4]:
        # Estimate P with  curr_N samples.
        estimated_P = np.zeros(shape=[Nr, f_len])
        for nr in range(Nr):
            print('Estimating:', nr)
            cov.estimate(x[nr][0:curr_N], f=f, p=p)
            estimated_P[nr][:] = cov['P']

        # Plot variance.
        mv.estimate(estimated_P)
        axarr[i_ax].plot(f, mv['var'])
        i_ax += 1

    # Label subplots & show
    axarr[0].set_title('var for N = {}'.format(N))
    axarr[1].set_title('var for N / 4 = {}'.format(N // 4))
    plt.show()

def model_order_selection(x, method='FPE', max_order=10):
    # Apply order selection on Covariance method.
    rand_index = randint(0, np.shape(x)[0] - 1)
    cov = CovarianceMethod()

    # Apply model order selection and plot results.
    mos = ModelOrderSelector()
    mos.apply(method, x[rand_index][:], max_order, cov)
    mos.plot()

def filter_and_autocorr(x):
    rand_index = randint(0, np.shape(x)[0] - 1)
    cov = CovarianceMethod()
    cov.estimate(x[rand_index][:], p=5)
    
    # Apply filter
    b = np.ndarray.flatten(cov['a'])
    a = [1]
    y = signal.lfilter(b, a, x[rand_index][:])

    # Plot result
    plt.figure()
    plt.plot(y)
    plt.title('Filtered signal')
    plt.xlabel('t [s]')
    plt.show()

    # Plot autocorrelation of the result
    autocorr = Autocorrelation()
    autocorr.estimate(y)
    autocorr.plot()

    # Plot autocorrelation on some segment
    r_yy = np.zeros(61)
    k = 0
    for i in np.arange(-30, 31):
        r_yy[k] = autocorr[i]
        k += 1
    plt.figure()
    plt.plot(np.arange(-30, 31), r_yy)
    plt.title('Autocorrelation of filtered signal')
    plt.xlabel('n')
    plt.show()

if __name__ == '__main__':
    # Read data
    N = 256
    Nr = 50
    x = read_data('data/data.csv', delimiter=',', file_size=[Nr, N])
    #plot_realisations(x, num=2)

    # 1. 2. 4. Apply various methods for spectral estimation
    apply_classical_methods(x)
    apply_parametric_methods(x)

    # 3. Apply window closing and show results
    window_closing_on_blackman_tukey(x)

    # 5. Apply FPE model order selection.
    model_order_selection(x, method='FPE', max_order=40)

    # 6. Filter sequence and show autocorrelation onf the result.
    filter_and_autocorr(x)

    # 7. Apply a few methods on all realisations
    apply_and_plot_all(x)

    # 8. Show estimated variance for Covariance method.
    show_variance_for_covariance_method(x, 10)

    # 9. Apply Covariance method with different orders.
    apply_and_plot_with_order(x, [N // 2, N // 4])
    