#' @title Different types of Entropy
#' @description Features from information theory point of view, i.e. different types of entropies.
#'
#'
#' @param x \code{vector} vector of accelerometry (or other signal similar in nature) signal within a window.
#' @param window The calculation needs the window size of the data. E.g window = 1 means each epoch is in one-minute window.
#'
#'
#' @importFrom DescTools entropy
#' @importFrom statcomp permutation_entropy
#' @importFrom TSEntropies SampEn_C
#'
#' @return A list with elements
#' \item{etp}{Entropy}
#' \item{perm_etp}{Permutation entropy}
#' \item{samp_etp}{sample entropy}
#'
#'
#' @export



info_feat = function(x){


  # entropy
  etp = Entropy(na.omit(x))
  perm_etp = permutation_entropy(ordinal_pattern_distribution(x,ndemb = 3))
  samp_etp = SampEn_C(x)

  ret = list("etp" = etp, "perm_etp" = perm_etp, "samp_etp" = samp_etp)
  return(ret)


}

