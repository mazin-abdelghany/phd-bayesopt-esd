# generate the penalty term
def feasibility_penalty(
        mu,
        power=0.8,
        alpha=0.05,
        beta_prime=0.7,
        alpha_prime=0.1):

    # calculate beta from power
    beta = 1-power

    # create the indicator functions
    def alpha_indicator(alpha_prime):
        if alpha_prime > alpha: return 1
        return 0

    def beta_indicator(beta_prime):
        if beta_prime > beta: return 1
        return 0

    return mu * (
        (((alpha_prime-alpha)/alpha)*alpha_indicator(alpha_prime)) + 
        (((beta_prime-beta)/power)*beta_indicator(beta_prime)) 
    )

# new feasibility penalty
def new_penalty(
        mu,
        power,
        alpha,
        alpha_prime,
        beta_prime):
    
    # calculate power from beta
    beta = 1-power
    
    if (alpha_prime > alpha) and (beta_prime > beta):
        return mu * ((alpha_prime - alpha)**2 + (beta_prime - beta)**2)
    return mu
    
# smooth feasibility penalty
def smooth_penalty(
        mu,
        power,
        alpha,
        alpha_prime,
        beta_prime,
        scaled=False,
        scale_factor=20):
    
    # calculate beta from power
    beta = 1-power
    
    # find the maximum alpha and beta
    if alpha <= 0.5:
        max_alpha = (alpha - 1)**2
    else:
        max_alpha = alpha**2
    
    if beta <= 0.5:
        max_beta = (beta - 1)**2
    else:
        max_beta = beta**2

    # min-max scale if scaled=True; note min alpha and beta is 0
    if scaled:
        val_a = (alpha_prime - alpha)**2
        val_b = (beta_prime - beta)**2

        val_a_scaled = val_a / max_alpha
        val_b_scaled = val_b / max_beta

        return scale_factor * (val_a_scaled + val_b_scaled)
    else:
        return mu * ((alpha_prime - alpha)**2 + (beta_prime - beta)**2)

# penalty with lower repulsion
def low_repulsion(
        mu,
        power,
        alpha,
        alpha_prime,
        beta_prime,
        scaled=False,
        scale_factor=20):
    
    # calculate beta from power
    beta = 1-power
    
    # find the maximum alpha and beta
    if alpha <= 0.5:
        max_alpha = (alpha - 1)**4
    else:
        max_alpha = alpha**4
    
    if beta <= 0.5:
        max_beta = (beta - 1)**4
    else:
        max_beta = beta**4

    # min-max scale if scaled=True; note min alpha and beta is 0
    if scaled:
        val_a = (alpha_prime - alpha)**4
        val_b = (beta_prime - beta)**4

        val_a_scaled = val_a / max_alpha
        val_b_scaled = val_b / max_beta

        return scale_factor * (val_a_scaled + val_b_scaled)
    else:
        return mu * ((alpha_prime - alpha)**4 + (beta_prime - beta)**4)

# step function penalty for alpha and beta and scaled loss for all components
def scaled_step(
        mu,
        power,
        alpha,
        alpha_prime,
        beta_prime,
        n_analyses,
        min_sample_size,
        max_sample_size,
        max_ess,
        alpha_epsilon = 0.005,
        beta_epsilon = 0.01,
        alpha_factor = 10,
        beta_factor = 10,
        max_ess_factor = 10):
    
    # calculate beta from power
    beta = 1-power

    alpha_met = (alpha_prime <= alpha) & ( (alpha - alpha_epsilon) <= alpha_prime )
    beta_met = (beta_prime <= beta) & ( (beta - beta_epsilon) <= beta_prime )

    loss_alpha = 0 if alpha_met else 1
    loss_beta = 0 if beta_met else 1

    loss_max_ess = (max_ess - min_sample_size) / ((n_analyses * max_sample_size) - min_sample_size)

    return (alpha_factor * loss_alpha) + (beta_factor * loss_beta) + (max_ess_factor * loss_max_ess)
