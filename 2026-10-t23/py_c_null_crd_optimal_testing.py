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
    asym = ctypes.CDLL("/tf/2026-10-t23/asym.so")
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
    return L, MatrixType, RowType


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

    result = asym.find_opt_asym_gst(
        alpha, delta, sigma, 1,
        num, limits, exits, 3, costs
    )
    return alpha, costs, delta, exits, limits, num, result, sigma


@app.cell
def _(costs, exits, limits):
    print("Optimal costs:", list(costs))
    print("Lower boundaries:", list(limits[1]))
    print("Upper boundaries:", list(limits[2]))
    print("Lower exit probabilities:", list(exits[1]))
    print("Upper exit probabilities:", list(exits[2]))
    return


@app.cell
def _(result):
    result
    return


@app.cell
def _(ctypes):
    py_asym = ctypes.CDLL("/tf/2026-10-t23/py_asym.so")
    return (py_asym,)


@app.cell
def _(RowType, ctypes, py_asym):
    py_asym.py_asym_driver.argtypes = (
        ctypes.POINTER(ctypes.c_double),  # alpha: double *
        ctypes.c_double,                  # delta: double
        ctypes.c_double,                  # sigma: double
        ctypes.POINTER(ctypes.c_double),  # num: double *
        ctypes.POINTER(RowType),          # limits: double [3][L]
        ctypes.POINTER(RowType),          # lbexits: double [3][L]
        ctypes.POINTER(RowType),          # ubexits: double [3][L]
        ctypes.POINTER(ctypes.c_double),  # objectives: double *
        ctypes.POINTER(ctypes.c_double),  # costs: double *
        ctypes.c_double,                  # lambda: double    
        ctypes.c_int,                     # obj: int
        ctypes.c_int,                     # K: int
    )

    py_asym.py_asym_driver.restype = ctypes.c_int
    return


@app.cell
def _(MatrixType, alpha, costs, ctypes, delta, limits, num, py_asym, sigma):
    # declare necessary inputs
    objectives = (ctypes.c_double * 12)(*([0.0] * 12))
    lbexits = MatrixType()
    ubexits = MatrixType()

    # Negative values request that the function choose starting costs
    costs_l = (ctypes.c_double * 2)(-1.0, -1.0)

    # lambda for L_lambda objective
    l_lambda = (ctypes.c_double)(0.5)

    result_l = py_asym.py_asym_driver(
        alpha,
        delta,
        sigma,
        num,
        limits,
        lbexits,
        ubexits,
        objectives,
        costs,
        l_lambda,
        6,
        3
    )
    return lbexits, ubexits


@app.cell
def _(costs, lbexits, limits, ubexits):
    print("Optimal costs:", list(costs))
    print("Lower boundaries:", list(limits[1]))
    print("Upper boundaries:", list(limits[2]))
    print("Efficacy alt:", list(lbexits[2]))
    print("Futility alt:", list(lbexits[1]))
    print("Cumu lower error:", list(lbexits[0]))
    print("Efficacy null:", list(ubexits[2]))
    print("Futility null:", list(ubexits[1]))
    print("Cumu upper error:", list(ubexits[0]))
    return


@app.cell
def _():
    from py_group_sequential_designs import generate_boundaries as bd

    return (bd,)


@app.cell
def _(bd):
    # null optimal design
    bd.calculate_bayes_dec_boundaries(
        n_analyses=3,
        alpha=0.05,
        beta=0.1,
        delta=1,
        variance=9,
        n_per_stage=64,
        obj_func=6,
        l_lambda=0
    )
    return


@app.cell
def _(bd):
    # calculate F2, should be same as below
    bd.calculate_bayes_dec_boundaries(
        n_analyses=3,
        alpha=0.05,
        beta=0.1,
        delta=1,
        variance=9,
        n_per_stage=64,
        obj_func=2
    )
    return


@app.cell
def _(bd):
    # with lambda = 0.5, F6 is same as F2
    bd.calculate_bayes_dec_boundaries(
        n_analyses=3,
        alpha=0.05,
        beta=0.1,
        delta=1,
        variance=9,
        n_per_stage=64,
        obj_func=6,
        l_lambda=0.5
    )
    return


if __name__ == "__main__":
    app.run()
