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

    return (plt,)


@app.cell
def _():
    from py_group_sequential_designs import simulate as sim
    from py_group_sequential_designs import sample_size as ss

    return sim, ss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Trial properties
    """)
    return


@app.cell
def _(mo):
    num_analyses = mo.ui.number(start=1, stop=20, label="Number of analyses")
    return (num_analyses,)


@app.cell
def _(num_analyses):
    num_analyses
    return


@app.cell
def _():
    target_alpha = 0.05
    target_power = 0.9
    delta0 = 0
    delta1 = 1.0
    sigma2 = 9.0
    return delta0, delta1, sigma2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Generate example bounds with sliders
    """)
    return


@app.cell
def _(mo):
    slider_or_value = mo.ui.switch(label="Slider or value")
    return (slider_or_value,)


@app.cell
def _(slider_or_value):
    if slider_or_value.value:
        label = "slider"
    else:
        label = "value"
    return (label,)


@app.cell
def _(label, mo, slider_or_value):
    mo.vstack([slider_or_value, mo.md(f"Has value: {label}")])
    return


@app.cell
def _(mo, num_analyses, slider_or_value):
    upper = []
    if slider_or_value.value:
        for i in range(num_analyses.value):
            upper.append(
                mo.ui.slider(
                    start=-10, stop=10, step=0.001,
                    label=f"Upper bound {i+1}",
                    orientation="horizontal",
                    show_value=True
                )
            )
    else:
        for i in range(num_analyses.value):
            upper.append(
                mo.ui.number(start=-10, stop=10, label="Number of analyses")
            )
    return (upper,)


@app.cell
def _(mo, num_analyses, slider_or_value):
    lower = []
    if slider_or_value.value:
        for j in range(num_analyses.value-1):
            lower.append(
                mo.ui.slider(
                    start=-10, stop=10, step=0.001,
                    label=f"Lower bound {j+1}",
                    orientation="horizontal",
                    show_value=True
                )
            )
    else:
        for j in range(num_analyses.value-1):
            lower.append(
                mo.ui.number(start=-10, stop=10, label="Number of analyses")
            )
    return (lower,)


@app.cell
def _(mo, upper):
    upper_array = mo.ui.array(upper)
    upper_array
    return (upper_array,)


@app.cell
def _(lower, mo):
    lower_array = mo.ui.array(lower)
    lower_array
    return (lower_array,)


@app.cell
def _(mo):
    sample_size = mo.ui.slider(start=1, stop=50, step=0.1, 
                                label="sample size", orientation="horizontal",
                                show_value=True)
    sample_size
    return (sample_size,)


@app.cell
def _(
    delta0,
    delta1,
    lower_array,
    num_analyses,
    sample_size,
    sigma2,
    sim,
    upper_array,
):
    sim.group_sequential_designs(
        n_analyses = num_analyses.value,
        upper_bounds = upper_array.value,
        lower_bounds = lower_array.value+ [upper_array.value[num_analyses.value-1]],
        n_patients = sample_size.value,
        null_hypothesis = delta0,
        alt_hypothesis = delta1,
        variance = sigma2
    )
    return


@app.cell
def _(delta0, lower_array, num_analyses, sample_size, sigma2, ss, upper_array):
    ss.max_ess(
        n_analyses = num_analyses.value,
        upper_bounds = upper_array.value,
        lower_bounds = lower_array.value+ [upper_array.value[num_analyses.value-1]],
        n_patients = sample_size.value,
        null_hypothesis = delta0,
        variance = sigma2
    )
    return


@app.cell
def _(lower_array, num_analyses, plt, upper_array):
    _fig, _ax = plt.subplots()

    _ax.plot(
        [k+1 for k in range(num_analyses.value)], 
        upper_array.value, 
        color = "red", 
        lw = 2
    )

    _ax.plot(
        [m+1 for m in range(num_analyses.value)], 
        lower_array.value + [upper_array.value[num_analyses.value-1]], 
        color = "red", 
        lw = 2
    )

    _ax.set_ylim(-4,3)

    _fig
    return


if __name__ == "__main__":
    app.run()
