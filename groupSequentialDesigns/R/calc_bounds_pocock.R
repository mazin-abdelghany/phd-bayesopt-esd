#' Calculate Pocock bounds for a group sequential clinical trial
#'
#' `calc_bounds_pocock` returns one- or two-sided boundary values for a group
#' sequential clinical trial design at a specified \eqn{\alpha} and sample
#' size using interval bisection.
#'
#' @param n_analyses Integer indicating the number of interim analyses
#' that will be performed in the group sequential design.
#' @param alpha Type I error rate to be targeted when calculating the bounds.
#' @param n_patients Vector of integers indicating the sample size (per group)
#' at each of the interim analyses. Note that the length of this vector must
#' correspond to the number of analyses `n_analyses`.
#' @param sided A string "one.sided" or "two.sided" indicating whether the
#' function returns one- or two-sided bounds. In one-sided bounds, the boundary
#' values are equal at the last analysis.
#' @returns A list of length three that contains the final upper and lower
#' bounds calculated and the last simulation run. The simulation output is from
#' `gsd_simulations` and shows the probability of stopping for futility and
#' efficacy under both the null and alternative hypotheses specified. It also
#' calculates the type I error (\eqn{\alpha}), the power (\eqn{1 - \beta}) where
#' \eqn{\beta} is the type II error, the expected sample size, and the variance
#' of the sample size.
#' @seealso [gsd_simulations()] for details on the simulation return.
#' @examples
#' calc_bounds_pocock(n_analyses = 3,
#'                    alpha = 0.05,
#'                    n_patients = c(20, 40, 60),
#'                    sided = "one.sided")

calc_bounds_pocock <- function(n_analyses = 3,
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

  # first alpha calculation
  sim <- gsd_simulations(n_analyses = n_analyses,
                         upper_bounds = ub1,
                         lower_bounds = lb1,
                         n_patients = n_patients)

  while ( abs(sim$alpha - alpha) > epsilon) {

    # calculate the first midpoint
    mid_u <- (ub1 + ub2) / 2
    mid_l <- -mid_u

    # calculate the simulated alpha
    sim <- gsd_simulations(n_analyses = n_analyses,
                           upper_bounds = mid_u,
                           lower_bounds = mid_l,
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
