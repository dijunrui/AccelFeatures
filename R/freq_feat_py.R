#' @title Frequency domain features Py
#' @description Features from frequency domain
#'
#'
#' @param x \code{vector} vector of accelerometry (or other signal similar in nature) signal within a window.
#' @param sampling_rate sampling frequency of the signal in Hz
#'
#'
#'
#' @importFrom reticulate source_python
#'
#' @return A list with elements
#' \item{dom_freq}{Dominant frequency}
#' \item{dom_freq_mag}{amplitude of dominant frequency}
#' \item{flatness}{spectral flatness}
#' \item{spec_entropy}{Shannon spectral entropy}
#'
#'
#' @export



freq_feat_py = function(x, sampling_rate,cutoff){

  xdt = data.frame(X = x)
  source_python("inst/domfreq.py")
  out = dominant_frequency(xdt,100,20,"X")
  return(out)

}

