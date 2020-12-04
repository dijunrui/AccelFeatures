#' @title Summary Statistics in Time Domain
#' @description Commonly used statistics in time domain
#'
#'
#' @param x \code{vector} vector of accelerometry (or other signal similar in nature) signal within a window.
#' @param r To calculate \code{ratio beyond r sigmas}, i.e. percent of samples that is r standard deviation from mean
#'
#'
#' @importFrom e1071 skewness kurtosis

#'
#' @return A list with elements
#' \item{mu}{mean}
#' \item{rms}{root mean squared, aka standard deviation}
#' \item{iqr}{inter quartile range}
#' \item{skw}{skewness}
#' \item{kurt}{kurtosis}
#' \item{rbrs}{ratio beyond r standardivation}
#'
#'
#' @export
#' @examples
#' count1 = c(t(example_activity_data$count[1,-c(1,2)]))
#' cos_coeff = ActCosinor(x = count1, window = 1)


summary_stat = function(x, r){


  n = length(x)

  # Summary stat
  mu = mean(x,na.rm = T)
  rms = sd(x,na.rm = T)
  iqr = IQR(x, na.rm = T)
  skw = skewness(x, na.rm = T)
  kurt = kurtosis(x, na.rm = T)
  rbrs = sum(abs(x - mu) > r * rms)/n

  ret = list("mu" = mu, "rms" = rms, "iqr" = iqr,
             "skw" = skw, "kurt" = kurt, "rbrs" = rbrs)

  return(ret)

}

