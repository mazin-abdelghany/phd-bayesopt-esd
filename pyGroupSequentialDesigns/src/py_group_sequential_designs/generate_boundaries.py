import numpy as np

# C shared object library imports and definitions for Bayes decision boundaries
import ctypes
from pathlib import Path

from . import simulate as sim

# Pocock boundaries
def calculate_pocock_boundaries(
        n_analyses=3,
        alpha=0.05,
        sided="one.sided"):
    
    # the precision of the estimate for alpha
    epsilon = 1e-8

    # starting first set of bounds
    ub1 = np.repeat(0, repeats = n_analyses)
    lb1 = -np.repeat(0, repeats = n_analyses)

    # starting second set of bounds
    ub2 = np.repeat(10, repeats = n_analyses)

    # first alpha calcuation
    sim_alpha, _, _ = sim.group_sequential_designs(
        n_analyses = n_analyses,
        upper_bounds = ub1,
        lower_bounds = lb1,
        n_patients = 10,
        null_hypothesis = 0,
        alt_hypothesis = 1,
        variance = 1
    )

    while abs(sim_alpha - alpha) > epsilon:

        # calculate the first midpoint
        mid_u = (ub1 + ub2) / 2
        mid_l = -mid_u

        # calculate the simulated alpha
        sim_alpha, _, _ = sim.group_sequential_designs(
            n_analyses = n_analyses,
            upper_bounds = mid_u,
            lower_bounds = mid_l,
            n_patients = 10,
            null_hypothesis = 0,
            alt_hypothesis = 1,
            variance = 1
        )

        if sim_alpha > alpha:
            ub1 = mid_u
            lb1 = mid_l
        else:
            ub2 = mid_u

    if sided == "one.sided":
        return [
            ub1,
            np.append(lb1[0:n_analyses-1], ub1[n_analyses-1])
        ]
    else:
        return [
            ub1,
            lb1
        ]

# O'Brien-Fleming boundaries
def calculate_of_boundaries(
        n_analyses=3,
        alpha=0.05,
        sided="one.sided"):

    # the precision of the estimate for alpha
    epsilon = 1e-8

    # starting first set of bounds
    ub1 = np.repeat(0, repeats = n_analyses)

    # starting second set of bounds
    ub2 = np.repeat(10, repeats = n_analyses)

    # of bounds
    of_u1 = ub1 * (np.arange(1, n_analyses+1, 1)/n_analyses)**(-0.5)
    of_l1 = -of_u1
    
    # first alpha calcuation
    sim_alpha, _, _ = sim.group_sequential_designs(
        n_analyses = n_analyses,
        upper_bounds = of_u1,
        lower_bounds = of_l1,
        n_patients = 10,
        null_hypothesis = 0,
        alt_hypothesis = 1,
        variance = 1
    )

    while abs(sim_alpha - alpha) > epsilon:

        # calculate the first midpoint
        mid_u = (ub1 + ub2) / 2
        mid_l = -mid_u

        # convert to O'Brien-Fleming bounds
        mid_of_u = mid_u * (np.arange(1, n_analyses+1, 1)/n_analyses)**(-0.5)
        mid_of_l = -mid_of_u

        # calculate the simulated alpha
        sim_alpha, _, _ = sim.group_sequential_designs(
            n_analyses = n_analyses,
            upper_bounds = mid_of_u,
            lower_bounds = mid_of_l,
            n_patients = 10,
            null_hypothesis = 0,
            alt_hypothesis = 1,
            variance = 1
        )

        if sim_alpha > alpha:
            ub1 = mid_u
        else:
            ub2 = mid_u

    if sided == "one.sided":
        return [
            mid_of_u,
            np.append(mid_of_l[0:n_analyses-1], mid_of_u[n_analyses-1]),
        ]
    else:
        return [
            mid_of_u,
            mid_of_l
        ]

# Triangular boundaries
def calculate_triangular_boundaries(
        n_analyses = 3,
        alpha = 0.05,
        delta = 0.5):

    # maximum information calculation
    I_L_term1 = (4 * (0.583**2))/n_analyses
    I_L_term2 = 8 * np.log(1/(2*alpha))
    I_L_term3 = (2*0.583) / np.sqrt(n_analyses)

    I_L = (np.sqrt(I_L_term1 + I_L_term2) - I_L_term3)**2 * (1/delta)**2

    # boundary calculation
    bounds_term1 = (2/delta) * np.log(1/(2*alpha))
    bounds_term2 = 0.583 * np.sqrt(I_L/n_analyses)

    analysis_fracs = np.arange(1, n_analyses+1, 1)/n_analyses

    I_L_fracs = I_L * analysis_fracs

    e_l = (bounds_term1 - bounds_term2 + ((0.25*delta) * analysis_fracs * I_L ))/np.sqrt(I_L_fracs)

    f_l = (-bounds_term1 + bounds_term2 + ((0.75*delta) * analysis_fracs * I_L ))/np.sqrt(I_L_fracs)

    return [
        e_l,
        f_l
    ]



# Variables and types that match the C code
L = 110

# In C, double limits[3][L] as a function parameter is double (*limits)[L]
RowType = ctypes.c_double * L
MatrixType = RowType * 3

# load the shared object file
py_asym_so_path = Path(__file__).resolve().parent / "py_asym.so"
py_asym = ctypes.CDLL(str(py_asym_so_path))

# input types to py_asym_driver
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

# return type
py_asym.py_asym_driver.restype = ctypes.c_int

# Bayes decision boundaries
def calculate_bayes_dec_boundaries(
        n_analyses,
        alpha,
        beta,
        delta,
        variance,
        n_per_stage,
        obj_func,
        l_lambda = None,
        return_all = False):

    if (obj_func == 6) and (l_lambda is None):
        raise ValueError("l_lambda must be set to calculate under F6")

    # convert delta H_0 = -delta, H_1 = delta to H_0 = 0 and H_1 = delta
    delta_prime = (ctypes.c_double)(delta / 2)

    # standard deviation on sum scale
    sigma_p = (ctypes.c_double)(np.sqrt(2 * variance))

    # alpha[0] = type II error (beta)
    # alpha[1] = type I error
    alpha_cinput = (ctypes.c_double * 2)(beta, alpha)

    # create array for number of patients input
    if L < n_analyses + 1:
        raise ValueError("L must be >= n_analyses + 1")
    
    values = (
        [0.0] 
        + [n_per_stage * (i+1) for i in range(n_analyses)] 
        + [0.0] * (L - (n_analyses + 1))
    )
    num = (ctypes.c_double * L)(*values)

    # declare necessary inputs
    limits = MatrixType()
    lbexits = MatrixType()
    ubexits = MatrixType()
    objectives = (ctypes.c_double * 12)(*([0.0] * 12))

    # Negative values request that the function choose starting costs
    costs = (ctypes.c_double * 2)(-1.0, -1.0)

    l_lambda = (ctypes.c_double)(l_lambda if l_lambda is not None else 0.5)
    
    obj_func_c = (ctypes.c_int)(obj_func)
    n_analyses_c = (ctypes.c_int)(n_analyses)

    info = py_asym.py_asym_driver(
        alpha_cinput,
        delta_prime,
        sigma_p,
        num,
        limits,
        lbexits,
        ubexits,
        objectives,
        costs,
        l_lambda,
        obj_func_c,
        n_analyses_c)

    # function to reparameterize to Z scale bounds
    def _sum_to_z_scale(
            sum_lower,
            sum_upper,
            n_per_stage,
            delta,
            sigma):

        # cumulative number of patients
        n = np.array([n_per_stage * (i+1) for i in range(n_analyses)])

        z_lower = (sum_lower + n * delta / 2) / (sigma * np.sqrt(n))
        z_upper = (sum_upper + n * delta / 2) / (sigma * np.sqrt(n))

        return [
            z_upper,
            z_lower
        ]

    sum_lower = list(limits[1])[1:n_analyses+1]
    sum_upper = list(limits[2])[1:n_analyses+1]
    
    bounds = _sum_to_z_scale(
        sum_lower = sum_lower,
        sum_upper = sum_upper,
        n_per_stage = n_per_stage,
        delta = delta,
        sigma = sigma_p
    )

    if return_all:
        return [
            bounds,
            sum_lower,
            sum_upper,
            info,
            list(costs)
        ]
    
    return bounds
