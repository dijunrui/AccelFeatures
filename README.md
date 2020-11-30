# AccelFeatures


An R package housing commonly used signal based feature extraction methods for machine learning algorithm development. It is built with IMU signals (accelerometer, gyroscope) in mind, however, many of the features will work for any signal.

The package contains the following features

## Summary Statististics

* __Mean__

* __standard deviation__ (a.k.a __root mean square__)

* __inter-quartile range__

* __skewness__

* __kurtosis__

* __ratio_beyond_r_sigma__: percent of observation that is outside r standard deviations


## Time series summary

* __Slope for linear trend__

* __autogressive coefficient__ lag 1

* __autocorrelation lag__ 1

* __mean cross rate__ 

## Information theory

* __entropy__

* __sample entropy__

* __permutation entropy__

## Frequency domain 

* __dominant frequency__

* __dominant frequency magnitude__

* __spectral flatness__

* __spectral entropy__

## Signal smoothness

* __jerk__

* __sparc__

* __dimensionless jerk__

* __log dimensionless jerk__
