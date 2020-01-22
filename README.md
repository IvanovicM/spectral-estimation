# Spectral Estimation
Various algorithms for spectral estimation. Based on the book "Modern Spectral Estimation - Theory &amp; Application", Steven M. Kay.

The methods that are implemented as part of this project are:

1. Classical Methods
    * Periodogram
    * Averaged Periodogram
    * Blackman-Tukey Method
    
2. Parametric Methods
    * Autocorrelation Method
    * Covariance Method
    * Modified Covariance Method
    * Burg Method
    
    
## How to run the tests?

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
