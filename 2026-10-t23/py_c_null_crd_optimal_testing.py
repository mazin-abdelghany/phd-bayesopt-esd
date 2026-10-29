import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import ctypes

    return (ctypes,)


@app.cell
def _(ctypes):
    asym = ctypes.CDLL("/tf/asym.so")
    return (asym,)


@app.cell
def _(asym, ctypes):
    # Matches L in the C code
    L = 110

    # In C, double limits[3][L] as a function parameter is double (*limits)[L]
    RowType = ctypes.c_double * L
    MatrixType = RowType * 3

    asym.find_opt_asym_gst.argtypes = (
        ctypes.POINTER(ctypes.c_double),  # alpha: double *
        ctypes.c_double,                  # delta: double
        ctypes.c_double,                  # sigma: double
        ctypes.c_int,                     # obj: int
        ctypes.POINTER(ctypes.c_double),  # num: double *
        ctypes.POINTER(RowType),          # limits: double limits[3][L]
        ctypes.POINTER(RowType),          # exits: double exits[3][L]
        ctypes.c_int,                     # K: int
        ctypes.POINTER(ctypes.c_double),  # costs: double *
    )

    asym.find_opt_asym_gst.restype = ctypes.c_int
    return L, MatrixType


@app.cell
def _(L, MatrixType, asym, ctypes):
    # alpha[0] = type II error (beta)
    # alpha[1] = type I error
    alpha = (ctypes.c_double * 2)(0.10, 0.05)

    delta = (ctypes.c_double)(0.5)
    sigma = (ctypes.c_double)(4.24264069)

    # cumulative group sizes
    # this follows the convention in the original C code
    # the first num is 0, the next 3 are the cumulative sample sizes
    # the last L - 4 values are 0
    num = (ctypes.c_double * L)(0.0, 64.0, 128.0, 192.0, *([0.0] * (L - 4)))

    # boundary limits and exit probabilities
    limits = MatrixType()
    exits = MatrixType()

    # Negative values request that the function choose starting costs
    costs = (ctypes.c_double * 2)(-1.0, -1.0)

    print(list(num))

    result = asym.find_opt_asym_gst(
        alpha, delta, sigma, 1,
        num, limits, exits, 3, costs
    )
    return costs, exits, limits


@app.cell
def _(costs, exits, limits):
    print("Optimal costs:", list(costs))
    print("Lower boundaries:", list(limits[1]))
    print("Upper boundaries:", list(limits[2]))
    print("Lower exit probabilities:", list(exits[1]))
    print("Upper exit probabilities:", list(exits[2]))
    return


if __name__ == "__main__":
    app.run()
