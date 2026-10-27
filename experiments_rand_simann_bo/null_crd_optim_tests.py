import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from py_group_sequential_designs import simulate as sim

    return np, sim


@app.cell
def _(np):
    n_analyses = 3
    n_patients = 64
    variance = 9
    delta = 1
    delta_prime = delta / 2

    lower_prime = np.array([-26.484171, -13.783852, 7.627324])
    upper_prime = np.array([ 41.738823,  29.038500,  7.627324])

    n = np.array([n_patients * (i + 1) for i in range(3)])

    sigma_p = np.sqrt(2 * variance)

    z_lower = (lower_prime + n * delta / 2) / (sigma_p * np.sqrt(n))
    z_upper = (upper_prime + n * delta / 2) / (sigma_p * np.sqrt(n))

    print(f"upper bounds: {z_upper}")
    print(f"lower bounds: {z_lower}")
    return delta, n_analyses, variance, z_lower, z_upper


@app.cell
def _(delta, n_analyses, sim, variance, z_lower, z_upper):
    sim.group_sequential_designs(
        n_analyses=n_analyses,
        lower_bounds=z_lower,
        upper_bounds=z_upper,
        n_patients=64,
        null_hypothesis=0,
        alt_hypothesis=delta,
        variance=variance,
        return_table=True
    )
    return


if __name__ == "__main__":
    app.run()
