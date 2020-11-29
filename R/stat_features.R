#' @title Simple Statistics/Time Domain Features
#' @description Commonly used statistics and time domain features.
#'
#'
#' @param x \code{vector} vector of dimension n*1440 which reprsents n days of 1440 minute activity data
#' @param window The calculation needs the window size of the data. E.g window = 1 means each epoch is in one-minute window.
#'
#'
#' @importFrom e1071 skewness kurtosis
#' @importFrom pracma linspace
#' @importFrom DescTools
#'
#' @return A list with elements
#' \item{mes}{MESOR which is short for midline statistics of rhythm, which is a rhythm adjusted mean. This represents mean activity level.}
#' \item{amp}{amplitude, a measure of half the extend of predictable variation within a cycle. This represents the highest activity one can achieve.}
#' \item{acro}{acrophase, a meaure of the time of the overall high values recurring in each cycle. Here it has a unit of radian. This represents time to reach the peak.}
#' \item{acrotime}{acrophase in the unit of the time (hours)}
#' \item{ndays}{Number of days modeled}
#'
#'
#' @references Cornelissen, G. Cosinor-based rhythmometry. Theor Biol Med Model 11, 16 (2014). https://doi.org/10.1186/1742-4682-11-16
#' @export
#' @examples
#' count1 = c(t(example_activity_data$count[1,-c(1,2)]))
#' cos_coeff = ActCosinor(x = count1, window = 1)


stat_features = function(x, sampling_rate){


  n = length(x)
  # Summary stat
  mu = mean(x,na.rm = T)
  rms = sd(x,na.rm = T)
  iqr = IQR(x, na.rm = T)
  skw = skewness(x, na.rm = T)
  kurt = kurtosis(x, na.rm = T)
  ratio_beyond_r_sig = sum(abs(x - mu) > r * rms)/n


  # linear trend
  t = linspace(0, n / sampling_rate, n)
  slope = lm(x~t)$coeffcoefficients[2]

  # autoregression and autocorrealtion
  ar1 = ar(x,aic = F, order.max = 1)$ar
  ac1 = acf(x,lag.max = 1,type = "correlation",plot = F)$acf[2,1,1]



  # mean cross rate
  mcr = sum(diff(sign(x - mu)) != 0)/n

  # entropy
  etp = Entropy(na.omit(x))
  perm_etp = permutation_entropy(ordinal_pattern_distribution(x,ndemb = 3))




}

