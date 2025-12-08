#' Title
#'
#' @param n_analyses
#' @param alpha
#' @param delta
#'
#' @returns
#' @export
#'
#' @examples

calc_bounds_triangular <- function(n_analyses = 3,
                                   alpha = 0.05,
                                   delta = 0.5) {

  I_L_term1 <- (4 * (0.583^2)) / n_analyses
  I_L_term2 <- 8 * log(1/(2*alpha))
  I_L_term3 <- (2 * 0.583) / sqrt(n_analyses)

  I_L <- (sqrt(I_L_term1 + I_L_term2) - I_L_term3)^2 * (1 / delta)^2

  bounds_term1 <- (2/delta) * log(1/(2*alpha))
  bounds_term2 <- 0.583 * sqrt(I_L / n_analyses)

  analysis_fracs <- (1:n_analyses / n_analyses)

  I_L_fracs <- I_L * analysis_fracs

  e_l <- (bounds_term1 - bounds_term2 + ((0.25*delta) * analysis_fracs * I_L )) / sqrt(I_L_fracs)

  f_l <- (-bounds_term1 + bounds_term2 + ((0.75*delta) * analysis_fracs * I_L )) / sqrt(I_L_fracs)

  return(list(upper_bounds = e_l, lower_bounds = f_l, info = I_L_fracs))

}
