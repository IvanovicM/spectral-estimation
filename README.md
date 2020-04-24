# Spectral Estimation
Various algorithms for spectral estimation. Based on the book "Modern Spectral Estimation - Theory &amp; Application", Steven M. Kay.

Methods implemented as part of this project:

1. Classical Methods
    * Periodogram
    * Averaged Periodogram
    * Blackman-Tukey Method
    
2. Parametric Methods
    * Autocorrelation Method
    * Covariance Method
    * Modified Covariance Method
    * Burg Method
    
## What's the difference between classical and parametric methods?
    
Classical spectral estimation methods are based on Fourier analysis.

On the other hand, it can be shown that if we know a model of a system, which produces the given signal by propagating white noise, we can estimate the signal’s PSD. Therefore, all parametric methods consist of choosing an appropriate model and estimating its parameters, along with substituting them into theoretical PSD expressions.

Results for one classical and one parametric method is shown below.

| <img src="images/avg_periodogram.png"> | <img src="images/covariance.png">|
|:---:|:---:|
| Average Periodogram Method Estimation | Covariance Method Estimation |

## How to choose the right parameters for these methods?
    
It's very important to choose right parameters, for as accurate as possible estimation.

Window closing is the method for determining a suitable window size for Blackman-Tukey spectral estimator. On the other hand, all of the prametric methods have a problem of determining the proper order of the model, p. Criterions that address this problem are FPE, AIC and CAT.

| <img src="images/bmt_window_closing.png"> | <img src="images/order_selection.png">|
|:---:|:---:|
| Window closing on Blackman-Tukey Method | Order Selection Illustrated |

# How to run the tests?

To run any test simply go to the directory above 'source' and type the following command in your terminal.

  ```shell
  python -m source.test.test_script
  ```

Test script can be any from the directory 'test':

- ```test_classical``` - To test all Classical Methods
- ```test_parametric``` - To test all Parametric Methods

If you want to test all implemented solutions for the tasks in the statement, simply go to the directory above 'source' and type:

```shell
  python -m source.test.task
  ```
