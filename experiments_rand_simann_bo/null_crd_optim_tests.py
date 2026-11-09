import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from py_group_sequential_designs import simulate as sim

    return np, sim


@app.cell
def _():
    # trial setup
    n_analyses = 3
    n_patients = 64
    variance = 9
    delta = 1
    delta_prime = delta / 2
    return delta, n_analyses, n_patients, variance


@app.cell
def _(np):
    # function to reparameterize to Z scale bounds
    def sum_to_z_scale(sum_lower, sum_upper, n_patients, delta, variance):
        # standard deviation on sum scale
        sigma_p = np.sqrt(2 * variance)

        # cumulative number of patients
        n = np.array([n_patients * (i + 1) for i in range(3)])

        z_lower = (sum_lower + n * delta / 2) / (sigma_p * np.sqrt(n))
        z_upper = (sum_upper + n * delta / 2) / (sigma_p * np.sqrt(n))

        return z_upper, z_lower

    return (sum_to_z_scale,)


@app.cell
def _(delta, n_patients, np, sum_to_z_scale, variance):
    # sum scale bounds
    lower_prime = np.array([-26.484171, -13.783852, 7.627324])
    upper_prime = np.array([ 41.738823,  29.038500,  7.627324])

    z_upper, z_lower = sum_to_z_scale(lower_prime, upper_prime, n_patients, delta, variance)

    print(f"upper bounds: {z_upper}")
    print(f"lower bounds: {z_lower}")
    return z_lower, z_upper


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


@app.cell
def _(delta, n_patients, sum_to_z_scale, variance):
    null_sum_upper = [54.5911657375491, 38.28427191201809, -4.160579188426703]
    null_sum_lower = [-20.543999715408056, -18.454512401183813, -4.160579188426703]

    null_upper, null_lower = sum_to_z_scale(null_sum_lower, null_sum_upper, n_patients, delta, variance)
    return null_lower, null_upper


@app.cell
def _(delta, n_analyses, null_lower, null_upper, sim, variance):
    sim.group_sequential_designs(
        n_analyses=n_analyses,
        lower_bounds=null_lower,
        upper_bounds=null_upper,
        n_patients=64,
        null_hypothesis=0,
        alt_hypothesis=delta,
        variance=variance,
        return_table=True
    )
    return


if __name__ == "__main__":
    app.run()
