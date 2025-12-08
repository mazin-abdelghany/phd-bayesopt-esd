#' Title
#'
#' @param n_analyses
#' @param upper_bounds
#' @param lower_bounds
#' @param n_patients
#' @param null_hypothesis
#' @param alt_hypothesis
#' @param variance
#'
#' @returns
#' @export
#'
#' @examples

gsd_simulations <- function(n_analyses = 3,
                            upper_bounds = c(2.5, 2, 1.5),
                            lower_bounds = c(0, 0.75, 1.5),
                            n_patients = c(20, 40, 60),
                            null_hypothesis = 0,
                            alt_hypothesis = 0.5,
                            variance = 1) {

  # sanity checks
  # sanity checks, function stops
  if(length(upper_bounds) != length(lower_bounds)) {
    stop("Warning: number of upper bounds must equal number of lower bounds")
  }

  if(length(n_patients) != length(upper_bounds)) {
    stop("Warning: number of patients vector must equal number of bounds")
  }

  if(n_analyses != length(upper_bounds)) {
    stop("Warning: number of analyses must equal number of bounds")
  }

  # assign values for null and alt hypotheses
  theta_0 <- null_hypothesis
  delta <- alt_hypothesis

  # empty mean vectors to fill
  mean_0 <- c()
  mean_1 <- c()

  # need to parse the upper and lower boundaries of the design
  # for futility and efficacy, must put the bounds of integration correctly
  # for pmvnorm
  futility_l_bounds <- list()
  futility_u_bounds <- list()
  efficacy_l_bounds <- list()
  efficacy_u_bounds <- list()

  n_analyses <- length(upper_bounds)

  for (i in 1:n_analyses) {

    # special case of i = 1
    if (i == 1) {
      futility_l_bounds[[i]] <- lower_bounds[i]
      futility_u_bounds[[i]] <- upper_bounds[i]
      efficacy_l_bounds[[i]] <- lower_bounds[i]
      efficacy_u_bounds[[i]] <- upper_bounds[i]
      next
    }

    # all other cases
    futility_l_bounds[[i]] <- c(lower_bounds[1:i-1], -Inf)
    futility_u_bounds[[i]] <- c(upper_bounds[1:i-1], lower_bounds[i])

    efficacy_l_bounds[[i]] <- c(lower_bounds[1:i-1], upper_bounds[i])
    efficacy_u_bounds[[i]] <- c(upper_bounds[1:i-1], Inf)
  }

  # list of probabilities to return
  probs_to_return <- list()

  # list of SIGMAs
  SIGMA_list <- list()

  for (i in 1:n_analyses) {
    if (i == 1) next

    # start with diagonal matrix for SIGMA
    SIGMA <- diag(nrow = i)

    # n = 2, need to fill all but 11, 22
    # n = 3, need to fill all but 11, 22, 33
    # n = 4, need to fill all but 11, 22, 33, 44
    # etc.
    for(i in 1:i) {
      for(j  in 1:i) {

        # leave the 1s on the diagonal, skip this iteration of for loop
        if(i == j) next

        # when i is less than j, the lower number of patients will be in numerator
        if(i < j) SIGMA[i,j] <- sqrt(n_patients[i] / n_patients[j])

        # when i is greater than j, the lower number of patients will be in numerator
        if(i > j) SIGMA[i,j] <- sqrt(n_patients[j] / n_patients[i])

      }
    }

    SIGMA_list[[i]] <- SIGMA
  }


  for (i in 1:n_analyses) {

    ##############
    # ANALYSIS 1 #
    ##############
    if(i == 1) {
      # mean under null
      mean_0[i] <- theta_0 * sqrt(n_patients[i]/(2*variance))

      # mean under alternative
      mean_1[i] <- delta * sqrt(n_patients[i]/(2*variance))

      # prob stop for futility, null
      futility_null <- pnorm(futility_l_bounds[[i]],
                             mean = mean_0,
                             sd = sqrt(variance))

      # prob stop for efficacy, null
      efficacy_null <- pnorm(efficacy_u_bounds[[i]],
                             mean = mean_0,
                             sd = sqrt(variance),
                             lower.tail = FALSE)

      # prob stop for futility, alt
      futility_alt <- pnorm(futility_l_bounds[[i]],
                            mean = mean_1,
                            sd = sqrt(variance))

      # prob stop for efficacy
      efficacy_alt <- pnorm(efficacy_u_bounds[[i]],
                            mean = mean_1,
                            sd = sqrt(variance),
                            lower.tail = FALSE)

      probs_to_return[[i]] <- c(futility_null, efficacy_null, futility_alt, efficacy_alt)
      names(probs_to_return[[i]]) <- c("futility_null", "efficacy_null", "futility_alt", "efficacy_alt")

      next
    }

    ######################
    # ALL OTHER ANALYSES #
    ######################

    # next mean under null
    mean_0[i] <- theta_0 * sqrt(n_patients[i] / (2 * variance))

    # next mean under alternative
    mean_1[i] <- delta * sqrt(n_patients[i]/ (2*variance))

    # bounds for these will be same
    # futility under null
    futility_null <- pmvnorm(lower = futility_l_bounds[[i]],
                             upper = futility_u_bounds[[i]],
                             mean = mean_0, corr = SIGMA_list[[i]])
    # futility under alt
    futility_alt <- pmvnorm(lower = futility_l_bounds[[i]],
                            upper = futility_u_bounds[[i]],
                            mean = mean_1, corr = SIGMA_list[[i]])

    # bounds for these will be same
    # futility under null
    efficacy_null <- pmvnorm(lower = efficacy_l_bounds[[i]],
                             upper = efficacy_u_bounds[[i]],
                             mean = mean_0, corr = SIGMA_list[[i]])
    # futility under alt
    efficacy_alt <- pmvnorm(lower = efficacy_l_bounds[[i]],
                            upper = efficacy_u_bounds[[i]],
                            mean = mean_1, corr = SIGMA_list[[i]])

    probs_to_return[[i]] <- c(futility_null, efficacy_null, futility_alt, efficacy_alt)
    names(probs_to_return[[i]]) <- c("futility_null", "efficacy_null", "futility_alt", "efficacy_alt")

  }

  # vector to collect the sum of futility and efficacy probabilities
  sum_probs <- c()

  # get alpha and power
  alpha <- 0
  power <- 0

  for (i in 1:n_analyses) {

    # pull the probabilities from the list
    tmp_probs <- probs_to_return[[i]]

    # gather them into a vector
    # 3:4 because we want to calculate under the alternative
    sum_probs <- c(sum_probs, sum(tmp_probs[3:4]))

    alpha <- tmp_probs[2] + alpha
    power <- tmp_probs[4] + power

  }

  # calculate the expected sample size
  ess <- sum(n_patients * sum_probs)
  vss <- sum(n_patients^2 * sum_probs) - ess^2

  # add the expected sample size to the list
  return_values <- append(probs_to_return, values = c(ess, vss, alpha, power))

  # name the list
  names_for_list <- as.vector(sapply("analysis_", paste0, 1:n_analyses))
  names_for_list <- c(names_for_list, "expected_sample_size", "var_sample_size",
                      "alpha", "power")
  names(return_values) <- names_for_list

  # return probabilities and ESS
  return_values
}
