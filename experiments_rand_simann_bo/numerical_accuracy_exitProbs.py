import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import mpmath as mp

    return (mp,)


@app.cell
def _(mp):
    def reference_gsd_probability(
        n_analyses,
        upper_bounds,
        lower_bounds,
        n_patients_analysis,
        means,
        stage,
        event="futility",
        dps=40,
    ):
        """
        Calculate the unconditional probability of futility or efficacy
        at one specified analysis, conditional on continuation through
        all previous analyses.

        Parameters
        ----------
        n_analyses : int
            Number of analyses.
        upper_bounds, lower_bounds : sequence
            Efficacy and futility boundaries at each analysis.
            Bounds may include -mp.inf and mp.inf.
        n_patients_analysis : sequence
            Cumulative information/patient counts at each analysis.
        means : sequence
            Mean of the standardized statistic at each analysis.
        stage : int
            Analysis number, using 1-based indexing.
        event : str
            "futility" or "efficacy".
        dps : int
            Decimal precision.

        Returns
        -------
        mp.mpf
            Unconditional probability of the specified event
            occurring at the specified stage.
        """

        if not 1 <= stage <= n_analyses:
            raise ValueError("stage must be between 1 and n_analyses")

        if event not in ("futility", "efficacy"):
            raise ValueError("event must be 'futility' or 'efficacy'")

        if len(upper_bounds) != n_analyses:
            raise ValueError("upper_bounds length must equal n_analyses")
        if len(lower_bounds) != n_analyses:
            raise ValueError("lower_bounds length must equal n_analyses")
        if len(n_patients_analysis) != n_analyses:
            raise ValueError("n_patients_analysis length must equal n_analyses")
        if len(means) != n_analyses:
            raise ValueError("means length must equal n_analyses")

        with mp.workdps(dps):
            upper = [mp.mpf(x) for x in upper_bounds]
            lower = [mp.mpf(x) for x in lower_bounds]
            info = [mp.mpf(x) for x in n_patients_analysis]
            mu = [mp.mpf(x) for x in means]

            sqrt2 = mp.sqrt(2)
            sqrt2pi = mp.sqrt(2 * mp.pi)

            def pdf(x):
                return mp.exp(-x**2 / 2) / sqrt2pi

            def cdf(x):
                return mp.erfc(-x / sqrt2) / 2

            def sf(x):
                return mp.erfc(x / sqrt2) / 2

            # Probability at stage 1 requires no integration.
            if stage == 1:
                if event == "futility":
                    return +cdf(lower[0] - mu[0])
                return +sf(upper[0] - mu[0])

            # Conditional distribution of Z_k given Z_(k-1).
            def conditional_params(k, z_prev):
                rho = mp.sqrt(info[k - 1] / info[k])
                cond_mean = mu[k] + rho * (z_prev - mu[k - 1])
                cond_sd = mp.sqrt(1 - rho**2)
                return cond_mean, cond_sd

            # Integrate over Z_1,...,Z_(stage-1), then evaluate
            # the final conditional event probability.
            def joint_integral(level, previous_z):
                if level == stage - 1:
                    z_prev = previous_z[-1]
                    cond_mean, cond_sd = conditional_params(
                        stage - 1, z_prev
                    )

                    if event == "futility":
                        return cdf((lower[stage - 1] - cond_mean) / cond_sd)
                    return sf((upper[stage - 1] - cond_mean) / cond_sd)

                def integrand(z):
                    if level == 0:
                        density = pdf(z - mu[0])
                    else:
                        cond_mean, cond_sd = conditional_params(
                            level, previous_z[-1]
                        )
                        density = (
                            pdf((z - cond_mean) / cond_sd) / cond_sd
                        )

                    return density * joint_integral(
                        level + 1, previous_z + [z]
                    )

                return mp.quad(
                    integrand,
                    [lower[level], upper[level]],
                )

            return +joint_integral(0, [])

    return (reference_gsd_probability,)


@app.cell
def _(mp, reference_gsd_probability):
    mp.mp.dps = 50

    upper = [3, 2, 1]
    lower = [-2, -1, 1]

    info = [20, 40, 60]
    means = [0, 0, 0]

    stage = 3

    # Probability of futility at stage 2
    p_fut_2 = reference_gsd_probability(
        3, upper, lower, info, means,
        stage=stage, event="futility"
    )

    # Probability of efficacy at stage 2
    p_eff_2 = reference_gsd_probability(
        3, upper, lower, info, means,
        stage=stage, event="efficacy"
    )

    print(f"Stage {stage} futility:", mp.nstr(p_fut_2, 30))
    print(f"Stage {stage} efficacy:", mp.nstr(p_eff_2, 30))
    return


if __name__ == "__main__":
    app.run()
