#' @title Trend features in time domain
#' @description Features depictingh trend of signals in the time domain
#'
#'
#' @param x \code{vector} vector of accelerometry (or other signal similar in nature) signal within a window.
#' @param sampling_rate sampling frequency of the signal in Hz
#'
#'
#' @importFrom pracma linspace
#'
#' @return A list with elements
#' \item{slope}{Slope for linear trend}
#' \item{ar1}{autoregressive coefficient with lag 1}
#' \item{ac1}{autocorrelation with lag 1}
#' \item{mcr}{mean cross rate, i.e. percent of samples crossing the mean}
#'
#'
#' @export
#' @examples
#' count1 = c(t(example_activity_data$count[1,-c(1,2)]))
#' cos_coeff = ActCosinor(x = count1, window = 1)


trend_feat = function(x, sampling_rate){


  n = length(x)

  # linear trend
  t = linspace(0, n / sampling_rate, n)
  slope = lm(x~t)$coeffcoefficients[2]

  # autoregression and autocorrealtion
  ar1 = ar(x,aic = F, order.max = 1)$ar
  ac1 = acf(x,lag.max = 1,type = "correlation",plot = F)$acf[2,1,1]

  # mean cross rate
  mcr = sum(diff(sign(x - mu)) != 0)/n

  ret = list("slope" = slope, "ar1" = ar1, "ac1" = ac1,
             "mcr" = mcr)

  return(ret)



}

