import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    return np, plt


@app.cell
def _():
    from py_group_sequential_designs import sample_size as ss
    from py_group_sequential_designs import simulate as sim
    from py_group_sequential_designs import feasibility_penalty as f

    return f, sim, ss


@app.cell
def _():
    # trial values
    n_analyses = 3
    upper_bounds = [2.703, 2.162, 1.622]
    lower_bounds = [0, 0.811, 1.622]
    n_patients = 26
    null_hypothesis = 0
    alt_hypothesis = 1
    variance = 9
    return lower_bounds, n_analyses, null_hypothesis, upper_bounds, variance


@app.cell
def _(np, variance):
    ess = np.empty(500)
    ess2 = np.empty(500)
    deltas = np.linspace(-2, variance**0.5*10, 500)
    return deltas, ess, ess2


@app.cell
def _(
    deltas,
    ess,
    lower_bounds,
    n_analyses,
    null_hypothesis,
    sim,
    upper_bounds,
    variance,
):
    for i, delta in enumerate(deltas):
        ess[i] = sim.group_sequential_designs(
            n_analyses=n_analyses,
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
            n_patients=26,
            null_hypothesis=null_hypothesis,
            alt_hypothesis=delta,
            variance=variance
        )[2]
    return


@app.cell
def _(deltas, ess2, n_analyses, null_hypothesis, sim, variance):
    for j, _delta in enumerate(deltas):
        ess2[j] = sim.group_sequential_designs(
            n_analyses=n_analyses,
            upper_bounds=[1.98, 1.98, 1.98],
            lower_bounds=[0,0,1.98],
            n_patients=28,
            null_hypothesis=null_hypothesis,
            alt_hypothesis=_delta,
            variance=variance
        )[2]
    return


@app.cell
def _(deltas, ess, ess2, plt):
    fig, ax = plt.subplots()
    ax.plot(deltas, ess, color = "blue")
    ax.plot(deltas, ess2, color = "red")
    ax.axhline(68, color = "green")
    ax.axhline(25.3)
    ax.axhline(27.5)
    ax.set_ylim(20, 70)
    plt.gca()
    return


@app.cell
def _(n_analyses, null_hypothesis, ss, variance):
    ss.max_ess(
        n_analyses=n_analyses,
            upper_bounds=[1.98, 1.98, 1.98],
            lower_bounds=[0,0,1.98],
            n_patients=28,
            null_hypothesis=null_hypothesis,
            variance=variance
    )
    return


@app.cell
def _(n_analyses, null_hypothesis, ss, variance):
    ss.max_ess(
        n_analyses=n_analyses,
    upper_bounds = [2.703, 2.162, 1.622],
    lower_bounds = [0, 0.811, 1.622], 
            n_patients=26,
            null_hypothesis=null_hypothesis,
            variance=variance
    )
    return


@app.cell
def _(n_analyses, null_hypothesis, sim, variance):
    sim.group_sequential_designs(
            n_analyses=n_analyses,
            upper_bounds=[1.98, 1.98, 1.98],
            lower_bounds=[0,0,1.98],
            n_patients=28,
            null_hypothesis=null_hypothesis,
            alt_hypothesis=10,
            variance=variance,
            return_table=True
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Standardised objective function

    The goal is for each component of the loss to contribute a equal amount to the total loss. A straighforward way to accomplish this is to have each component min-max scaled so that the loss value is in the interval [0,1]. Then a scaling factor can be applied in order to modify the magnitude of the loss.

    $$
    \mathcal{L}(\alpha, \beta, \mathbb{E}[N]) = \lambda\mathcal{L(\alpha)} + \gamma\mathcal{L(\beta)} + \phi\mathbb{E}[N\mid\delta^\star]
    $$

    $\mathcal{L}(\cdot)$ can be min-max scaled as such:

    $$
    \mathcal{L}_{scaled} = \frac{\mathcal{L} - \min(\mathcal{L})}{\max(\mathcal{L})-\min(\mathcal{L})}
    $$

    Because of the importance of ensuring $\alpha$ and $\beta$ are within target, the loss for each of these will be set as a step function:

    $$
    \mathcal{L}(\alpha)=\begin{cases}
    0, \quad \alpha^* - \epsilon_1 < \alpha' < \alpha^* \\
    1, \quad \text{otherwise.}
    \end{cases}
    $$

    This guarantees that the maximum loss for type I and II error is 1 and the minimum is 0; no min-max scaling is required.

    There is no transformation of the loss for the expected sample size by any function. The maximum value that the maximum expected sample size can take occurs when the study design requires that all $K$ stages are completed and $n$ is at the maximum boundary value set by the search space. Similarly, the minimum value that the maximum expected sample size can take occurs when the study design on requires that the first stage be completed and $n$ is at the minimum boudary value set by the search space. The scaling is then done as such:

    $$
    \mathbb{E}[N\mid\delta^\star]_{scaled} = \frac{\mathbb{E}[N\mid\delta^\star] - n}{Kn-n}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The first loss to test

    The first loss to test would have $\lambda=\gamma=\phi=1$:

    $$
    \mathcal{L}(\alpha, \beta, \mathbb{E}[N]) = \mathcal{L(\alpha)} + \mathcal{L(\beta)} + \mathbb{E}[N\mid\delta^\star]_{scaled}
    $$
    where
    $$
    \mathcal{L}(\alpha)=\begin{cases}
    0, \quad \alpha^* - \epsilon_1 < \alpha' < \alpha^* \\
    1, \quad \text{otherwise.}
    \end{cases}
    %
    \qquad\qquad
    %
    \mathcal{L}(\beta)=\begin{cases}
    0, \quad \beta^* - \epsilon_2 < \beta' < \beta^* \\
    1, \quad \text{otherwise.}
    \end{cases}
    $$
    and
    $$
    \mathbb{E}[N\mid\delta^\star]_{scaled} = \frac{\mathbb{E}[N\mid\delta^\star] - n_{min}}{Kn_{max}-n_{min}}
    $$
    where $n_{min}$ and $n_{max}$ are the boundaries of the search space for the sample size dimention.
    """)
    return


@app.cell
def _(f):
    f.scaled_step(
        mu=150,
        power=0.9,
        alpha=0.05,
        alpha_prime=0.2,
        beta_prime=0.2,
        n_analyses=3,
        min_sample_size=20,
        max_sample_size=160,
        max_ess=55,
        alpha_factor=0,
        beta_factor=0,
        max_ess_factor=1
    )
    return


if __name__ == "__main__":
    app.run()
