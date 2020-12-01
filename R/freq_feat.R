#' @title Frequency domain features
#' @description Features from frequency domain
#'
#'
#' @param x \code{vector} vector of accelerometry (or other signal similar in nature) signal within a window.
#' @param sampling_rate sampling frequency of the signal in Hz
#'
#'
#'
#' @importFrom seewave meanspec sfm sh
#' @importFrom pracma linspace
#' @importFrom DescTools entropy
#' @importFrom statcomp permutation_entropy
#' @importFrom TSEntropies SampEn_C
#'
#' @return A list with elements
#' \item{dom_freq}{Dominant frequency}
#' \item{dom_freq_mag}{amplitude of dominant frequency}
#' \item{flatness}{spectral flatness}
#' \item{spec_entropy}{Shannon spectral entropy}
#'
#'
#' @export
#' @examples
#' count1 = c(t(example_activity_data$count[1,-c(1,2)]))
#' cos_coeff = ActCosinor(x = count1, window = 1)


freq_feat = function(x, sampling_rate){


  spec_x = meanspec(x, f = sampling_rate, plot = F,norm = F)
  dom_freq_mag = max(spec_x[,2])
  dom_freq = spec_x[which.max(spec_x[,2]),1] * 1000
  flatness =  sfm(spec_x)
  spec_entropy = sh(spec_x)


  ret = list("dom_freq" = dom_freq, "dom_freq_mag" = dom_freq_mag, "flatness" = flatness,
             "spec_entropy" = spec_entropy)

  return(ret)

}

