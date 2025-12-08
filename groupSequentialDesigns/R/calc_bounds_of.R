#' Title
#'
#' @param n_analyses
#' @param alpha
#' @param n_patients
#' @param sided
#'
#' @returns
#' @export
#'
#' @examples

calc_bounds_of <- function(n_analyses = 3,
                           alpha = 0.05,
                           n_patients = c(20, 40, 60),
                           sided = "one.sided") {

  # the precision of the estimate for alpha
  epsilon <- 1e-8

  # starting upper bounds
  ub1 <- rep(0, length.out = n_analyses)
  lb1 <- -ub1

  # starting lower bounds
  ub2 <- rep(10, length.out = n_analyses)
  lb2 <- -ub2

  # ob bounds
  ob_u1 <- ub1 * (1:n_analyses / n_analyses)^-0.5
  ob_l1 <- -ob_u1

  ob_u2 <- ub2 * (1:n_analyses / n_analyses)^-0.5
  ob_l2 <- -ob_u2

  # first alpha calculation
  sim <- gsd_simulations(n_analyses = n_analyses,
                         upper_bounds = ob_u1,
                         lower_bounds = ob_l1,
                         n_patients = n_patients)

  while ( abs(sim$alpha - alpha) > epsilon) {

    # calculate the first midpoint
    mid_u <- (ub1 + ub2) / 2
    mid_l <- -mid_u

    # get ob bounds
    mid_ob_u <- mid_u * (1:n_analyses / n_analyses)^-0.5
    mid_ob_l <- -mid_ob_u

    # calculate the simulated alpha
    sim <- gsd_simulations(n_analyses = n_analyses,
                           upper_bounds = mid_ob_u,
                           lower_bounds = mid_ob_l,
                           n_patients = n_patients)

    #
    if (sim$alpha > alpha) {

      ub1 <- mid_u
      lb1 <- mid_l

    } else {

      ub2 <- mid_u
      lb2 <- mid_l

    }

  }

  if (sided == "one.sided") {

    # return the values of interest
    return( list(upper_bounds = mid_ob_u,
                 lower_bounds = c(mid_ob_l[1:n_analyses-1], mid_ob_u[n_analyses]),
                 simulation = sim) )

  } else {

    # return the values of interest
    return( list(upper_bounds = mid_ob_u,
                 lower_bounds = mid_ob_l,
                 simulation = sim) )

  }

}
