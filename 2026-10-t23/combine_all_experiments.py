import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    return np, pd, plt


@app.cell
def _():
    import os

    return (os,)


@app.cell
def _():
    from py_group_sequential_designs import generate_boundaries as bd
    from py_group_sequential_designs import simulate as sim
    from py_group_sequential_designs import sample_size as ss

    return bd, ss


@app.cell
def _():
    bayes_opt_dir = "/tf/experiments_rand_simann_bo/bayes_opt_experiments/"
    return (bayes_opt_dir,)


@app.cell
def _(bayes_opt_dir, os):
    data_files = os.listdir(bayes_opt_dir)
    data_files.sort()
    return (data_files,)


@app.cell
def _(data_files):
    keys = []
    for i, file in enumerate(data_files):
        if file.endswith(".csv") and file.startswith("3-stage"):
            try: 
                keys.append("3-stage_" + file.split("haltons_")[1])
            except IndexError:
                keys.append("3-stage_" + file.split("_loss_")[1])
        elif file.endswith(".csv") and file.startswith("3-stage") and ("no_n" in file):
            keys.append("5-stage_" + file.split("_loss_")[1])
        elif file.endswith(".csv") and file.startswith("5-stage"):
            try: 
                keys.append("5-stage_" + file.split("haltons_")[1])
            except IndexError:
                keys.append("5-stage_" + file.split("_loss_")[1])
    return (keys,)


@app.cell
def _(bayes_opt_dir, data_files, keys, pd):
    bo_data_3_stage = pd.DataFrame(data=None)
    bo_data_5_stage = pd.DataFrame(data=None)

    for j, key in enumerate(keys):
        if "DS_Store" in data_files[j]: continue

        if "3-stage" in data_files[j]:
            tmp_df = pd.read_csv(filepath_or_buffer=bayes_opt_dir+data_files[j])
            tmp_df["method"] = key
            bo_data_3_stage = pd.concat([bo_data_3_stage, tmp_df], ignore_index=True)
        if "5-stage" in data_files[j]:
            tmp_df = pd.read_csv(filepath_or_buffer=bayes_opt_dir+data_files[j])
            tmp_df["method"] = key
            bo_data_5_stage = pd.concat([bo_data_5_stage, tmp_df], ignore_index=True)
        
    bo_data_3_stage.drop(columns="Unnamed: 0", inplace=True)
    bo_data_5_stage.drop(columns="Unnamed: 0", inplace=True)
    return bo_data_3_stage, bo_data_5_stage


@app.cell
def _(bo_data_3_stage):
    alpha_mask_3_stage = (0.0495 <= bo_data_3_stage["alpha"]) & (bo_data_3_stage["alpha"] <= 0.0505)
    power_mask_3_stage = (0.899 <= bo_data_3_stage["power"]) & (bo_data_3_stage["power"] <= 0.91)
    mESS_mask_3_stage = (100 <= bo_data_3_stage["max_ess"]) & (bo_data_3_stage["max_ess"] <= 130)
    return alpha_mask_3_stage, mESS_mask_3_stage, power_mask_3_stage


@app.cell
def _(bo_data_5_stage):
    alpha_mask_5_stage = (0.0495 <= bo_data_5_stage["alpha"]) & (bo_data_5_stage["alpha"] <= 0.0505)
    power_mask_5_stage = (0.899 <= bo_data_5_stage["power"]) & (bo_data_5_stage["power"] <= 0.91)
    mESS_mask_5_stage = (100 <= bo_data_5_stage["max_ess"]) & (bo_data_5_stage["max_ess"] <= 130)
    return alpha_mask_5_stage, mESS_mask_5_stage, power_mask_5_stage


@app.cell
def _(
    alpha_mask_3_stage,
    alpha_mask_5_stage,
    bo_data_3_stage,
    bo_data_5_stage,
    mESS_mask_3_stage,
    mESS_mask_5_stage,
    power_mask_3_stage,
    power_mask_5_stage,
):
    best_bo_designs_3_stage = bo_data_3_stage[alpha_mask_3_stage & power_mask_3_stage & mESS_mask_3_stage]
    best_bo_designs_5_stage = bo_data_5_stage[alpha_mask_5_stage & power_mask_5_stage & mESS_mask_5_stage]
    return best_bo_designs_3_stage, best_bo_designs_5_stage


@app.cell
def _(bd):
    tri_3_stage = bd.calculate_triangular_boundaries(n_analyses=3)
    return (tri_3_stage,)


@app.cell
def _(ss, tri_3_stage):
    tri_3_stage_n = ss.find_sample_size(
        n_analyses=3,
        upper_bounds=tri_3_stage[0],
        lower_bounds=tri_3_stage[1],
        null_hypothesis=0,
        alt_hypothesis=1,
        variance=9
    )
    return (tri_3_stage_n,)


@app.cell
def _(ss, tri_3_stage, tri_3_stage_n):
    tri_max_ess_3_stage = ss.max_ess(
        n_analyses=3,
        upper_bounds=tri_3_stage[0],
        lower_bounds=tri_3_stage[1],
        null_hypothesis=0,
        n_patients=tri_3_stage_n[0],
        variance=9
    )
    return (tri_max_ess_3_stage,)


@app.cell
def _(best_bo_designs_3_stage, np, plt, tri_3_stage, tri_max_ess_3_stage):
    plt.plot(
        [i for i in range(3)],
        [best_bo_designs_3_stage.loc[
            best_bo_designs_3_stage["max_ess"].idxmin()
        ]["upper1"], best_bo_designs_3_stage.loc[
            best_bo_designs_3_stage["max_ess"].idxmin()
        ]["upper2"], best_bo_designs_3_stage.loc[
            best_bo_designs_3_stage["max_ess"].idxmin()
        ]["upper3"]]
    )

    plt.plot(
        [i for i in range(3)],
        [best_bo_designs_3_stage.loc[
            best_bo_designs_3_stage["max_ess"].idxmin()
        ]["lower1"], best_bo_designs_3_stage.loc[
            best_bo_designs_3_stage["max_ess"].idxmin()
        ]["lower2"], best_bo_designs_3_stage.loc[
            best_bo_designs_3_stage["max_ess"].idxmin()
        ]["upper3"]]
    )

    plt.text(1,0.2, "bayes opt mESS = " + str(np.round(best_bo_designs_3_stage.loc[
            best_bo_designs_3_stage["max_ess"].idxmin()
        ]["max_ess"],2)))

    plt.text(1,0, "tri mESS = " + str(np.round(tri_max_ess_3_stage,2)))

    plt.plot([i for i in range(3)], tri_3_stage[0])
    plt.plot([i for i in range(3)], tri_3_stage[1])
    return


@app.cell
def _(bd):
    tri_5_stage = bd.calculate_triangular_boundaries(n_analyses=5)
    return (tri_5_stage,)


@app.cell
def _(ss, tri_5_stage):
    tri_5_stage_n = ss.find_sample_size(
        n_analyses=5,
        upper_bounds=tri_5_stage[0],
        lower_bounds=tri_5_stage[1],
        null_hypothesis=0,
        alt_hypothesis=1,
        variance=9
    )
    return (tri_5_stage_n,)


@app.cell
def _(ss, tri_5_stage, tri_5_stage_n):
    tri_max_ess_5_stage = ss.max_ess(
        n_analyses=5,
        upper_bounds=tri_5_stage[0],
        lower_bounds=tri_5_stage[1],
        null_hypothesis=0,
        n_patients=tri_5_stage_n[0],
        variance=9
    )
    return (tri_max_ess_5_stage,)


@app.cell
def _(best_bo_designs_5_stage, np, plt, tri_5_stage, tri_max_ess_5_stage):
    plt.plot(
        [i for i in range(5)],
        [best_bo_designs_5_stage.loc[
            best_bo_designs_5_stage["max_ess"].idxmin()
        ]["upper1"], best_bo_designs_5_stage.loc[
            best_bo_designs_5_stage["max_ess"].idxmin()
        ]["upper2"], best_bo_designs_5_stage.loc[
            best_bo_designs_5_stage["max_ess"].idxmin()
        ]["upper3"], best_bo_designs_5_stage.loc[
            best_bo_designs_5_stage["max_ess"].idxmin()
        ]["upper4"], best_bo_designs_5_stage.loc[
            best_bo_designs_5_stage["max_ess"].idxmin()
        ]["upper5"]]
    )

    plt.plot(
        [i for i in range(5)],
        [best_bo_designs_5_stage.loc[
            best_bo_designs_5_stage["max_ess"].idxmin()
        ]["lower1"], best_bo_designs_5_stage.loc[
            best_bo_designs_5_stage["max_ess"].idxmin()
        ]["lower2"], best_bo_designs_5_stage.loc[
            best_bo_designs_5_stage["max_ess"].idxmin()
        ]["lower3"], best_bo_designs_5_stage.loc[
            best_bo_designs_5_stage["max_ess"].idxmin()
        ]["lower4"], best_bo_designs_5_stage.loc[
            best_bo_designs_5_stage["max_ess"].idxmin()
        ]["upper5"]]
    )

    plt.text(1,-1.4, "bayes opt mESS = " + str(np.round(best_bo_designs_5_stage.loc[
            best_bo_designs_5_stage["max_ess"].idxmin()
        ]["max_ess"],2)))

    plt.text(1,-1,"tri mESS = " + str(np.round(tri_max_ess_5_stage,2)))

    plt.plot([i for i in range(5)], tri_5_stage[0], color="purple")
    plt.plot([i for i in range(5)], tri_5_stage[1], color="purple")
    return


@app.cell
def _():
    rand_dir = "/tf/experiments_rand_simann_bo/random_search_experiments/"
    return


@app.cell
def _():
    sim_ann_dir = "/tf/experiments_rand_simann_bo/simulated_annealing_experiments/"
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
