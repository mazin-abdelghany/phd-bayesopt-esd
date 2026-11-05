import marimo

__generated_with = "0.25.2"
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

    return bd, sim, ss


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
def _(np, plt, tri_3_stage, tri_max_ess_3_stage):
    def plot_best_design(best_design, method):
        plt.plot(
            [i for i in range(3)],
            [
                best_design.loc[best_design["max_ess"].idxmin()][f"upper{i}"]
                for i in range(1,4)
            ]
        )

        plt.plot(
            [i for i in range(3)],
            [
                best_design.loc[best_design["max_ess"].idxmin()][f"lower{i}"]
                for i in range(1,3)
            ] + [best_design.loc[best_design["max_ess"].idxmin()]["upper3"]]
        )

        plt.text(0.6,0.4, method + " method = " + best_design.loc[
                best_design["max_ess"].idxmin()
            ]["method"])

        plt.text(1,0.2, method + " mESS = " + str(np.round(best_design.loc[
                best_design["max_ess"].idxmin()
            ]["max_ess"],2)))

        plt.text(1,0, "tri mESS = " + str(np.round(tri_max_ess_3_stage,2)))

        plt.plot([i for i in range(3)], tri_3_stage[0], color = "purple")
        plt.plot([i for i in range(3)], tri_3_stage[1], color = "purple")

        plt.show()

    return (plot_best_design,)


@app.cell
def _(best_bo_designs_3_stage, plot_best_design):
    # filtering by max_ess gives equivalent design as filtering by obj_func
    plot_best_design(best_bo_designs_3_stage, "bayes opt")
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
def _(np, plt, tri_5_stage, tri_max_ess_5_stage):
    def plot_best_design5(best_design, method):
        plt.plot(
            [i for i in range(5)],
            [
                best_design.loc[best_design["max_ess"].idxmin()][f"upper{i}"]
                for i in range(1, 6)
            ]
        )

        plt.plot(
            [i for i in range(5)],
            [
                best_design.loc[best_design["max_ess"].idxmin()][f"lower{i}"]
                for i in range(1, 5)
            ] + [best_design.loc[best_design["max_ess"].idxmin()]["upper5"]]
        )

        plt.text(1,-0.4, method + " method = " + best_design.loc[
                best_design["max_ess"].idxmin()
            ]["method"])

        plt.text(1,-1.4, method + " mESS = " + str(np.round(best_design.loc[
                best_design["max_ess"].idxmin()
            ]["max_ess"],2)))

        plt.text(1,-1,"tri mESS = " + str(np.round(tri_max_ess_5_stage,2)))

        plt.plot([i for i in range(5)], tri_5_stage[0], color="purple")
        plt.plot([i for i in range(5)], tri_5_stage[1], color="purple")

        plt.show()

    return (plot_best_design5,)


@app.cell
def _(best_bo_designs_5_stage, plot_best_design5):
    plot_best_design5(best_bo_designs_5_stage, "bayes opt")
    return


@app.cell
def _():
    rand_dir = "/tf/experiments_rand_simann_bo/random_search_experiments/"
    return (rand_dir,)


@app.cell
def _(os, rand_dir):
    rand_files = os.listdir(rand_dir)
    rand_files.sort()
    return (rand_files,)


@app.cell
def _(pd, rand_dir, rand_files):
    rand_5_stage = pd.DataFrame(data=None)
    rand_3_stage = pd.DataFrame(data=None)

    for f in rand_files:
        if "DS_" in f: continue
        if f.endswith(".csv") and f.startswith("3"):
            _tmp_df = pd.read_csv(filepath_or_buffer=rand_dir+f)
            _tmp_df["method"] = f.split("_")[-1]
            rand_3_stage = pd.concat([rand_3_stage, _tmp_df], ignore_index=True)
        elif f.endswith(".csv") and f.startswith("5"):
            rand_5_stage = pd.read_csv(filepath_or_buffer=rand_dir + f)
            rand_5_stage["method"] = "25x1200"
    return f, rand_3_stage, rand_5_stage


@app.cell
def _(rand_3_stage):
    alpha_mask_rand_3_stage = (0.0495 <= rand_3_stage["alpha"]) & (rand_3_stage["alpha"] <= 0.0505)
    power_mask_rand_3_stage = (0.899 <= rand_3_stage["power"]) & (rand_3_stage["power"] <= 0.91)
    mESS_mask_rand_3_stage = (100 <= rand_3_stage["max_ess"]) & (rand_3_stage["max_ess"] <= 130)
    return alpha_mask_rand_3_stage, power_mask_rand_3_stage


@app.cell
def _(alpha_mask_rand_3_stage, power_mask_rand_3_stage, rand_3_stage):
    best_rand_designs_3_stage = rand_3_stage[
        alpha_mask_rand_3_stage & power_mask_rand_3_stage #& mESS_mask_rand_3_stage
    ]
    return (best_rand_designs_3_stage,)


@app.cell
def _(best_rand_designs_3_stage, plot_best_design):
    plot_best_design(best_rand_designs_3_stage, "rand")
    return


@app.cell
def _(rand_5_stage):
    alpha_mask_rand_5_stage = (0.0495 <= rand_5_stage["alpha"]) & (rand_5_stage["alpha"] <= 0.0508)
    power_mask_rand_5_stage = (0.899 <= rand_5_stage["power"]) & (rand_5_stage["power"] <= 0.91)
    mESS_mask_rand_5_stage = (100 <= rand_5_stage["max_ess"]) & (rand_5_stage["max_ess"] <= 130)
    return alpha_mask_rand_5_stage, power_mask_rand_5_stage


@app.cell
def _(alpha_mask_rand_5_stage, power_mask_rand_5_stage, rand_5_stage):
    best_rand_designs_5_stage = rand_5_stage[
        alpha_mask_rand_5_stage & power_mask_rand_5_stage #& mESS_mask_rand_5_stage
    ]
    return (best_rand_designs_5_stage,)


@app.cell
def _(best_rand_designs_5_stage):
    best_rand_designs_5_stage
    return


@app.cell
def _(best_rand_designs_5_stage, plot_best_design5):
    plot_best_design5(best_rand_designs_5_stage, "rand")
    return


@app.cell
def _():
    sim_ann_dir = "/tf/experiments_rand_simann_bo/simulated_annealing_experiments/"
    return (sim_ann_dir,)


@app.cell
def _(os, sim_ann_dir):
    sim_ann_files = os.listdir(sim_ann_dir)
    sim_ann_files.sort()
    return (sim_ann_files,)


@app.cell
def _(f, pd, sim_ann_dir, sim_ann_files):
    sim_ann_5_stage = pd.DataFrame(data=None)
    sim_ann_3_stage = pd.DataFrame(data=None)

    for sf in sim_ann_files:
        if "DS_" in f: continue
        if sf.endswith(".csv") and sf.startswith("3"):
            _tmp_df = pd.read_csv(filepath_or_buffer=sim_ann_dir + sf)
            _tmp_df["method"] = sf.split("_")[-1]
            sim_ann_3_stage = pd.concat([sim_ann_3_stage, _tmp_df], ignore_index=True)
        elif sf.endswith(".csv") and sf.startswith("5"):
            sim_ann_5_stage = pd.read_csv(filepath_or_buffer=sim_ann_dir + sf)
            sim_ann_5_stage["method"] = "25x1200"
    return sim_ann_3_stage, sim_ann_5_stage


@app.cell
def _(sim_ann_3_stage):
    alpha_mask_sim_ann_3_stage = (0.0495 <= sim_ann_3_stage["alpha"]) & (sim_ann_3_stage["alpha"] <= 0.0505)
    power_mask_sim_ann_3_stage = (0.899 <= sim_ann_3_stage["power"]) & (sim_ann_3_stage["power"] <= 0.91)
    mESS_mask_sim_ann_3_stage = (100 <= sim_ann_3_stage["max_ess"]) & (sim_ann_3_stage["max_ess"] <= 136)
    return (
        alpha_mask_sim_ann_3_stage,
        mESS_mask_sim_ann_3_stage,
        power_mask_sim_ann_3_stage,
    )


@app.cell
def _(
    alpha_mask_sim_ann_3_stage,
    mESS_mask_sim_ann_3_stage,
    power_mask_sim_ann_3_stage,
    sim_ann_3_stage,
):
    best_sim_ann_designs_3_stage = sim_ann_3_stage[
        alpha_mask_sim_ann_3_stage & power_mask_sim_ann_3_stage & mESS_mask_sim_ann_3_stage
    ]
    return (best_sim_ann_designs_3_stage,)


@app.cell
def _(best_sim_ann_designs_3_stage):
    best_sim_ann_designs_3_stage
    return


@app.cell
def _(best_sim_ann_designs_3_stage, plot_best_design):
    plot_best_design(best_sim_ann_designs_3_stage, "sim ann")
    return


@app.cell
def _(sim_ann_5_stage):
    alpha_mask_sim_ann_5_stage = (0.0495 <= sim_ann_5_stage["alpha"]) & (sim_ann_5_stage["alpha"] <= 0.0508)
    power_mask_sim_ann_5_stage = (0.899 <= sim_ann_5_stage["power"]) & (sim_ann_5_stage["power"] <= 0.91)
    mESS_mask_sim_ann_5_stage = (100 <= sim_ann_5_stage["max_ess"]) & (sim_ann_5_stage["max_ess"] <= 130)
    return alpha_mask_sim_ann_5_stage, power_mask_sim_ann_5_stage


@app.cell
def _(alpha_mask_sim_ann_5_stage, power_mask_sim_ann_5_stage, sim_ann_5_stage):
    best_sim_ann_designs_5_stage = sim_ann_5_stage[
        alpha_mask_sim_ann_5_stage & power_mask_sim_ann_5_stage #& mESS_mask_sim_ann_5_stage
    ]
    return (best_sim_ann_designs_5_stage,)


@app.cell
def _(best_sim_ann_designs_5_stage):
    best_sim_ann_designs_5_stage
    return


@app.cell
def _(best_sim_ann_designs_5_stage, plot_best_design5):
    plot_best_design5(best_sim_ann_designs_5_stage, "sim ann")
    return


@app.cell
def _(best_sim_ann_designs_5_stage):
    best_sim_ann_designs_5_stage.loc[best_sim_ann_designs_5_stage["max_ess"].idxmin()]
    return


@app.cell
def _(sim):
    sim.group_sequential_designs(
        n_analyses=5,
        upper_bounds=[9.3047356154463, 7.21987721850268, 4.314560451229432,4.1135991195456425,1.6472959479516802],
        lower_bounds=[-6.741779960186249, -5.26607048137944, -2.0000448356109084,-0.550440254392748,1.6472959479516802],
        n_patients=20, null_hypothesis=0,
        alt_hypothesis=0.5, variance=1,
        return_table=True, grid_points=5001
    )
    return


@app.cell
def _(sim):
    sim.group_sequential_designs(
        n_analyses=5,
        upper_bounds=[10, 7.21987721850268, 4.314560451229432,4.1135991195456425,1.6472959479516802],
        lower_bounds=[-6.741779960186249, -5.26607048137944, -2.0000448356109084,-0.550440254392748,1.6472959479516802],
        n_patients=20, null_hypothesis=0,
        alt_hypothesis=0.5, variance=1,
        return_table=True, grid_points=5001
    )
    return


@app.cell
def _(sim):
    sim.group_sequential_designs(
        n_analyses=5,
        upper_bounds=[7.21987721850268, 7.21987721850268, 4.314560451229432,4.1135991195456425,1.6472959479516802],
        lower_bounds=[-6.741779960186249, -5.26607048137944, -2.0000448356109084,-0.550440254392748,1.6472959479516802],
        n_patients=20, null_hypothesis=0,
        alt_hypothesis=0.5, variance=1,
        return_table=True, grid_points=201
    )
    return


if __name__ == "__main__":
    app.run()
