import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    return mpl, np, pd, plt


@app.cell
def _():
    from py_group_sequential_designs import generate_boundaries as bd
    from py_group_sequential_designs import simulate as sim
    from py_group_sequential_designs import sample_size as ss

    return bd, sim


@app.cell
def _():
    info_dict = {
        0 : "improper input parameters",
        1 : "success",
        2 : "reached or exceeded maxfev",
        3 : "xtol too small, no further improvement possible",
        4 : "poor progress, last five jacobian evaluations",
        5 : "poor progress, last ten iterations",
        6 : "success, near window",
        7 : "success, bad residual",
        8 : "success, near windown and bad residual"
    }
    return (info_dict,)


@app.cell
def _():
    # trial setup
    null = 0
    n_analyses = 3
    alpha = 0.05
    beta = 0.1
    delta = 1
    variance = 9

    # bayes decision objective function and l_lambda
    obj_func = 6
    return alpha, beta, delta, n_analyses, null, obj_func, variance


@app.cell
def _(np):
    n_per_stage = np.linspace(45, 155, 111)
    rgb_x = np.linspace(0, 1, 111)

    deltas = np.linspace(0, 1, 200)
    return deltas, n_per_stage, rgb_x


@app.cell
def _(bd, delta, n_analyses):
    tri = bd.calculate_triangular_boundaries(
        n_analyses=n_analyses,
        alpha=0.05,
        delta=delta
    )
    return (tri,)


@app.cell
def _(
    alpha,
    bd,
    beta,
    delta,
    deltas,
    info_dict,
    mpl,
    n_analyses,
    n_per_stage,
    np,
    null,
    obj_func,
    plt,
    rgb_x,
    sim,
    variance,
):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,8), layout="constrained")
    first_few_fig, first_few_ax = plt.subplots(nrows=2, ncols=1, figsize=(12,8), layout="constrained")

    rgb_array = mpl.colormaps["PuOr"](rgb_x)[:, :3]

    failed_n_lambda0 = []
    failed_n_lambda1 = []

    failed_reason_lambda0 = []
    failed_reason_lambda1 = []

    min_ess_null = 1e6
    min_ess_alt = 1e6

    min_ess_null_n = None
    min_ess_alt_n = None

    for i, n in enumerate(n_per_stage):
        power0 = None
        power1 = None

        cost_and_bounds_lambda0 = bd.calculate_bayes_dec_boundaries(
            n_analyses,
            alpha,
            beta,
            delta,
            variance,
            n,
            obj_func,
            l_lambda = 0,
            return_costs=True
        )

        cost_and_bounds_lambda1 = bd.calculate_bayes_dec_boundaries(
            n_analyses,
            alpha,
            beta,
            delta,
            variance,
            n,
            obj_func,
            l_lambda = 1,
            return_costs=True
        )

        bounds_lambda0 = cost_and_bounds_lambda0[0]
        bounds_lambda1 = cost_and_bounds_lambda1[0]

        fail_indicator_lambda0 = cost_and_bounds_lambda0[3]
        fail_indicator_lambda1 = cost_and_bounds_lambda1[3]

        # Record failures 
        if fail_indicator_lambda0 != 1:
            print("failed at lambda = 0!")
            failed_n_lambda0.append(n)
            failed_reason_lambda0.append(info_dict[fail_indicator_lambda0])

        if fail_indicator_lambda1 != 1:
            print("failed at lambda = 1!")
            failed_n_lambda1.append(n)
            failed_reason_lambda1.append(info_dict[fail_indicator_lambda1])


        ess_alt_lambda0 = np.empty(len(deltas))
        ess_alt_lambda1 = np.empty(len(deltas))

        # Simulate each successful design 
        for j, d in enumerate(deltas):

            if fail_indicator_lambda0 == 1:
                simu_lambda0 = sim.group_sequential_designs(
                    n_analyses=n_analyses,
                    upper_bounds=bounds_lambda0[0],
                    lower_bounds=bounds_lambda0[1],
                    n_patients=n,
                    null_hypothesis=null,
                    alt_hypothesis=d,
                    variance=variance
                )

                ess_alt_lambda0[j] = simu_lambda0[3]

                if d == 1:
                    power0 = simu_lambda0[1]

            if fail_indicator_lambda1 == 1:
                simu_lambda1 = sim.group_sequential_designs(
                    n_analyses=n_analyses,
                    upper_bounds=bounds_lambda1[0],
                    lower_bounds=bounds_lambda1[1],
                    n_patients=n,
                    null_hypothesis=null,
                    alt_hypothesis=d,
                    variance=variance
                )

                ess_alt_lambda1[j] = simu_lambda1[3]

                if d == 1:
                    power1 = simu_lambda1[1]

        if (fail_indicator_lambda1 == 1) and (min_ess_null > ess_alt_lambda0[0]):
            min_ess_null = ess_alt_lambda0[0]
            min_ess_null_n = n
        
        if (fail_indicator_lambda1 == 1) and (min_ess_alt > ess_alt_lambda1[-1]):
            min_ess_alt = ess_alt_lambda1[-1]
            min_ess_alt_n = n

        # Plot successful designs 
        if (fail_indicator_lambda0 == 1) and (power0 is not None) and (power0 > 0.8999):
            ax[0].plot(deltas, ess_alt_lambda0, c=tuple(rgb_array[i]))

        if (fail_indicator_lambda1 == 1) and (power1 is not None) and (power1 > 0.8999):
            ax[1].plot(deltas, ess_alt_lambda1, c=tuple(rgb_array[i]))

        # plot first few designs to assess differences
        if (fail_indicator_lambda0 == 1) and (power0 is not None) and (power0 > 0.8999) and (i < 19):
            first_few_ax[0].plot(deltas, ess_alt_lambda0, c=tuple(rgb_array[i]))
            first_few_ax[0].text(1.01, ess_alt_lambda0[-1]-1, str(n))

        if (fail_indicator_lambda1 == 1) and (power1 is not None) and (power1 > 0.8999) and (i < 19):
            first_few_ax[1].plot(deltas, ess_alt_lambda1, c=tuple(rgb_array[i]))
            first_few_ax[1].text(-0.04, ess_alt_lambda1[0]-1, str(n))

        # find the minimum ess at null / alternative and the n at which this is achieved
    
    return (
        ax,
        failed_n_lambda0,
        failed_n_lambda1,
        failed_reason_lambda0,
        failed_reason_lambda1,
        fig,
        first_few_ax,
        first_few_fig,
        min_ess_alt,
        min_ess_alt_n,
        min_ess_null,
        min_ess_null_n,
        rgb_array,
    )


@app.cell
def _(ax, fig, mpl):
    sm = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=52, vmax=152),
        cmap="PuOr"
    )

    fig.colorbar(sm, ax=ax, label="Per stage sample size")
    return (sm,)


@app.cell
def _(deltas, n_analyses, null, sim, tri, variance):
    tri_ess = []
    for m, l in enumerate(deltas):
        tri_ess.append(sim.group_sequential_designs(
            n_analyses=n_analyses,
            upper_bounds=tri[0],
            lower_bounds=tri[1],
            n_patients=64,
            null_hypothesis=null,
            alt_hypothesis=l,
            variance=variance
        )[3])
    return (tri_ess,)


@app.cell(expand_output=True)
def _(ax, deltas, fig, tri_ess):
    ax[0].set_title("ESS under alt across deltas - Bayes null optimal")
    ax[0].set_xlabel("Delta")
    ax[0].set_ylabel("Expected sample size under alt")

    ax[0].plot(deltas, tri_ess, color = "red")

    ax[1].set_title("ESS under alt across deltas - Bayes alt optimal")
    ax[1].set_xlabel("Delta")
    ax[1].set_ylabel("Expected sample size under alt")

    ax[1].plot(deltas, tri_ess, color = "red")

    fig
    return


@app.cell(expand_output=True)
def _(deltas, first_few_ax, first_few_fig, tri_ess):
    first_few_ax[0].set_title("ESS under alt across deltas - Bayes null optimal")
    first_few_ax[0].set_xlabel("Delta")
    first_few_ax[0].set_ylabel("Expected sample size under alt")

    first_few_ax[0].plot(deltas, tri_ess, color = "red")

    first_few_ax[1].set_title("ESS under alt across deltas - Bayes alt optimal")
    first_few_ax[1].set_xlabel("Delta")
    first_few_ax[1].set_ylabel("Expected sample size under alt")

    first_few_ax[1].plot(deltas, tri_ess, color = "red")

    first_few_fig
    return


@app.cell
def _(min_ess_alt, min_ess_alt_n):
    print(min_ess_alt)
    print(min_ess_alt_n)
    return


@app.cell
def _(min_ess_null, min_ess_null_n):
    print(min_ess_null)
    print(min_ess_null_n)
    return


@app.cell
def _(
    failed_n_lambda0,
    failed_n_lambda1,
    failed_reason_lambda0,
    failed_reason_lambda1,
    pd,
):
    failures_lambda0 = pd.DataFrame(
        {
            "n" : failed_n_lambda0,
            "reason" : failed_reason_lambda0
        }
    )

    failures_lambda1 = pd.DataFrame(
        {
            "n" : failed_n_lambda1,
            "reason" : failed_reason_lambda1
        }
    )
    return failures_lambda0, failures_lambda1


@app.cell
def _(failures_lambda0):
    failures_lambda0
    return


@app.cell
def _(failures_lambda1):
    failures_lambda1
    return


@app.cell
def _(bd, delta, deltas, n_analyses, np, null, plt, rgb_array, sim, variance):
    fig2, ax2 = plt.subplots(figsize=(12,6))

    for i1, n1 in enumerate(np.arange(52,136)):
        bounds1 = bd.calculate_triangular_boundaries(
            n_analyses=n_analyses,
            alpha=0.05,
            delta=delta
        )

        ess1 = []

        # varying delta (the alternative)
        for j1, d1 in enumerate(deltas):
            ess1.append(sim.group_sequential_designs(
                n_analyses=n_analyses,
                upper_bounds=bounds1[0],
                lower_bounds=bounds1[1],
                n_patients=n1,
                null_hypothesis=null,
                alt_hypothesis=d1,
                variance=variance
            )[3])

        ax2.plot(deltas, ess1, c=tuple(rgb_array[i1]))
    return ax2, fig2


@app.cell
def _(ax2, fig2, sm):
    fig2.colorbar(sm, ax=ax2, label="Per stage sample size")

    ax2.set_title("ESS under alternative across deltas - Triangular")
    ax2.set_xlabel("Delta")
    ax2.set_ylabel("Expected sample size under alt")

    fig2
    return


@app.cell
def _():
    test_n = 61
    return (test_n,)


@app.cell
def _(alpha, bd, beta, delta, n_analyses, obj_func, test_n, variance):
    bbb = bd.calculate_bayes_dec_boundaries(
        n_analyses,
        alpha,
        beta,
        delta,
        variance,
        test_n,
        obj_func,
        l_lambda=1
    )
    return (bbb,)


@app.cell
def _(bbb, delta, n_analyses, null, sim, test_n, variance):
    sim.group_sequential_designs(
        n_analyses=n_analyses,
        upper_bounds=bbb[0],
        lower_bounds=bbb[1],
        n_patients=test_n,
        null_hypothesis=null,
        alt_hypothesis=delta,
        variance=variance,
        return_table=True
    )
    return


@app.cell
def _(alpha, bd, beta, delta, n_analyses, obj_func, test_n, variance):
    bounds = bd.calculate_bayes_dec_boundaries(
        n_analyses,
        alpha,
        beta,
        delta,
        variance,
        test_n,
        obj_func,
        l_lambda=0
    )
    return (bounds,)


@app.cell
def _(bounds):
    bounds
    return


@app.cell
def _(bounds, delta, n_analyses, null, sim, test_n, variance):
    sim.group_sequential_designs(
        n_analyses=n_analyses,
        upper_bounds=bounds[0],
        lower_bounds=bounds[1],
        n_patients=test_n,
        null_hypothesis=null,
        alt_hypothesis=delta,
        variance=variance,
        return_table=True
    )
    return


@app.cell
def _(bounds, n_analyses, plt, tri):
    plt.plot([x+1 for x in range(n_analyses)], bounds[0], color = "purple")
    plt.plot([x+1 for x in range(n_analyses)], bounds[1], color = "purple")

    plt.plot([x+1 for x in range(n_analyses)], tri[0], color = "orange")
    plt.plot([x+1 for x in range(n_analyses)], tri[1], color = "orange")
    return


@app.cell
def _(delta, n_analyses, null, sim, tri, variance):
    sim.group_sequential_designs(
        n_analyses=n_analyses,
        upper_bounds=tri[0],
        lower_bounds=tri[1],
        n_patients=64,
        null_hypothesis=null,
        alt_hypothesis=delta,
        variance=variance,
        return_table=True
    )
    return


if __name__ == "__main__":
    app.run()
