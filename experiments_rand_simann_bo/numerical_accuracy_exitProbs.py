import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    need to install mpmath
    uv pip install mpmath
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    testing the bounds
        const bounds = Bounds(3) {
            .upper = .{3, 2, 1},
            .lower = .{-2, -1, 1},
        };


    the below calculation is for futility under the null at the last stage.
    """)
    return


@app.cell
def _():
    import mpmath as mp

    def reference_probability(dps=40):
        with mp.workdps(dps):
            sqrt2 = mp.sqrt(2)
            rho12 = mp.sqrt(mp.mpf(1) / 2)
            rho23 = mp.sqrt(mp.mpf(2) / 3)

            def Phi(x):
                return mp.erfc(-x / sqrt2) / 2

            def conditional_z3_cdf(z2):
                return Phi((1 - rho23 * z2) / mp.sqrt(1 - rho23**2))

            def integrand_z2(z2):
                # P(Z3 <= 1 | Z2=z2) * f(Z2 | Z1=z1)
                # The integration over Z1 is performed outside.
                return conditional_z3_cdf(z2)

            def inner(z1):
                mean2 = rho12 * z1
                sd2 = mp.sqrt(1 - rho12**2)

                def f_z2(z2):
                    density = mp.exp(
                        -((z2 - mean2) / sd2)**2 / 2
                    ) / (sd2 * mp.sqrt(2 * mp.pi))

                    return density * integrand_z2(z2)

                return mp.quad(f_z2, [-1, 2])

            def outer(z1):
                density1 = mp.exp(-z1**2 / 2) / mp.sqrt(2 * mp.pi)
                return density1 * inner(z1)

            result = mp.quad(outer, [-2, 3])
            return +result


    for digits in (25, 40):
        p = reference_probability(digits)
        print(f"dps={digits}: {mp.nstr(p, 30)}")
    return


app._unparsable_cell(
    r"""
    501: 0.67694139 49931851
    1001: 0.67694139500 58686
    10_001: 0.67694139500671 78
    20_001: 0.67694139500671 29
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    500: 0.67694 01241014688
    1000: 0.676941 0779156123
    10_000: 0.67694139 18415071
    100_000: 0.6769413949 750639
    """,
    name="_"
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
