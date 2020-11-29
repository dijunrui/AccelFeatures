# AccelFeatures


An R package housing commonly used signal based feature extraction methods for machine learning algorithm development. It is built with IMU signals (accelerometer, gyroscope) in mind, however, many of the features will work for any signal.

The package contains the following features

## Summary Statististics

* mean

* standard deviation (a.k.a root mean square)

* inter-quartile range

* skewness

* kurtosis

* ratio_beyond_r_sigma


## Time series summary

* Slope for linear trend

* autogressive coefficient lag 1

* autocorrelation lag 1

* mean cross rate

## Information theory

* entropy

* permutation entropy

## Frequency domain 

* dominant frequency

* dominant frequency magnitude

* spectral flatness

* spectral entropy

## Signal smoothness

* jerk

* sparc

* dimensionless jerk

* log dimensionless jerk
