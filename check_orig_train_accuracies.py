import glob
import matplotlib
from matplotlib.patches import Ellipse 
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import natsort
import numpy as np
import os

from scipy import stats
from scipy.spatial import ConvexHull
import seaborn as sns
import pandas as pd




# Ersetzen Sie Ihren rcParams Block mit diesem:
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 20,           # Vergrößert von 16
    "axes.labelsize": 20,      # Vergrößert von 16
    "axes.titlesize": 22,      # Vergrößert von 16
    "xtick.labelsize": 18,     # Vergrößert von 16
    "ytick.labelsize": 18,     # Vergrößert von 16
    "legend.fontsize": 18,     # Vergrößert von 16
    "figure.titlesize": 24     # Vergrößert von 20
})

tasks = ["sst2", "mrpc", "qqp", "rte"]
# tasks = ["mnli"]                     maxloss_list.append(np.load(os.path.join(f"data/testOrig_2/v2-intrinsic-maxloss-accuracies-{task}_{lin_type}-small-{seed}.npy" )))
all_task = {}

for task in tasks:
    all_median = {}
    for lin_type in ["linear", "nonlinear"]:
        path = f"data/results_max_loss/intrinsic/ABv2-intrinsic-maxloss-accuracies-{task}_{lin_type}"
        path_ = f"data/lin/models/testOrig_2/v2-intrinsic-maxloss-accuracies-{task}"
        all_median[lin_type] = {}
        for model in ["multiberts","electra", "roberta"]:#, "electra"
            maxloss_list = []
            if model.startswith('electra'):
                maxloss_list.append(np.load(os.path.join(f"{path}-small-{model}-None.npy" )))
                model_short =  'E'
                seeds = [0]  # Electra has only one seed
            elif model.startswith('roberta'):
                maxloss_list.append(np.load(os.path.join(f"{path}-small-{model}-None.npy" )))
                model_short = 'R'
                seeds = [0]
            elif model.startswith('multiberts'):
                seeds = range(24)
                for seed in seeds:
                    maxloss_list.append(np.load(os.path.join(f"{path}-small-{model}-{seed}.npy" )))

            maxloss_list = np.array(maxloss_list)
            intrinsic_distances = []
            for seed in seeds:
                if model.startswith('multiberts'):
                    model_short = f"M{int(seed)}"
                max_indices = np.argmin(maxloss_list[seed][:, :, -1], axis=1)
                intrinsic_distances.append(
                    np.insert(maxloss_list[seed][np.arange(maxloss_list[seed].shape[0]), max_indices][:,-1], seed, 0)
                )
                all_median[lin_type][model_short] = np.median(np.insert(maxloss_list[seed][np.arange(maxloss_list[seed].shape[0]), max_indices][:,-1], seed, 0))
                print("median  for model", {lin_type}, model_short, ":", all_median[lin_type][model_short])
            intrinsic_distances = np.array(intrinsic_distances)

            all_median[lin_type][model_short] = np.median(intrinsic_distances)
            print(f"Intrinsic distances for task {task}: {intrinsic_distances}")
    all_task[task] = all_median

# Gemeinsamer Plot für alle Tasks
fig, axes = plt.subplots(1, 4, figsize=(12, 7.5), sharey=False)
axes = axes.flatten()

for ax, task in zip(axes, tasks):
    data_linear = all_task[task]["linear"]
    data_nonlinear = all_task[task]["nonlinear"]

    x_linear = list(data_linear.keys())
    y_linear = list(data_linear.values())
    x_nonlinear = list(data_nonlinear.keys())
    y_nonlinear = list(data_nonlinear.values())

    # Gleiche Reihenfolge für beide Typen
    assert x_linear == x_nonlinear
    x_keys_orig = x_linear  # Ursprüngliche Reihenfolge
    
    # Neue Reihenfolge: RoBERTa -> MultiBERT Seeds -> Electra
    x_keys_reordered = []
    y_linear_reordered = []
    y_nonlinear_reordered = []
    
    # Erst RoBERTa
    for i, key in enumerate(x_keys_orig):
        if key == "R":
            x_keys_reordered.append(key)
            y_linear_reordered.append(y_linear[i])
            y_nonlinear_reordered.append(y_nonlinear[i])
    
    # Dann MultiBERT Seeds (sortiert)
    multibert_items = [(key, y_linear[i], y_nonlinear[i]) for i, key in enumerate(x_keys_orig) if key.startswith("M")]
    multibert_items.sort(key=lambda x: int(x[0][1:]))  # Sortiert nach Seed-Nummer
    for key, yl, ynl in multibert_items:
        x_keys_reordered.append(key)
        y_linear_reordered.append(yl)
        y_nonlinear_reordered.append(ynl)
    
    # Zuletzt Electra
    for i, key in enumerate(x_keys_orig):
        if key == "E":
            x_keys_reordered.append(key)
            y_linear_reordered.append(y_linear[i])
            y_nonlinear_reordered.append(y_nonlinear[i])

    x_ticks = np.arange(len(x_keys_reordered))

    # Scatter-Punkte zeichnen mit neuer Reihenfolge
    ax.scatter(x_ticks, y_linear_reordered, color='tab:blue', label='Linear', marker='o')
    ax.scatter(x_ticks, y_nonlinear_reordered, color='tab:red', label='Nonlinear', marker='o')

    # Gruppierte xticklabels: Ein Label pro Modellfamilie
    tick_pos = []
    tick_labels = []

    # RoBERTa (ganz links)
    r_indices = [i for i, k in enumerate(x_keys_reordered) if k == "R"]
    if r_indices:
        tick_pos.append(np.mean(r_indices))
        tick_labels.append(r"$\mathcal{R}$")

    # MultiBERTs (Mitte)
    m_indices = [i for i, k in enumerate(x_keys_reordered) if k.startswith("M")]
    if m_indices:
        tick_pos.append(np.mean(m_indices))
        tick_labels.append(r"$\rightarrow \mathcal{M}$")

    # Electra (ganz rechts)
    e_indices = [i for i, k in enumerate(x_keys_reordered) if k == "E"]
    if e_indices:
        tick_pos.append(np.mean(e_indices))
        tick_labels.append(r"$\mathcal{E}$")

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=0)

    ax.set_title(f"{task.upper()}")
    ax.set_xlabel("Target Encoder")
    ax.set_ylabel("Max train loss")

# Gemeinsame Legende rechts neben den Plots
handles, labels = axes[0].get_legend_handles_labels()
axes[-1].legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.35))
plt.tight_layout()
plt.suptitle("Max Training Loss per Task", fontsize=24, y=1.02)
plt.subplots_adjust(top=0.85)  # Mehr Platz oben für die Legende
plt.savefig(f"New_all_in_one_scatter_praes.png", dpi=300, bbox_inches="tight")
plt.show()

exit()

lin_types = ["nonlinear", "linear"]
tasks = ["cola", "mrpc", "qnli", "qqp", "sst2", "rte"]

# Farben für Tasks
colors = sns.color_palette("tab10", n_colors=len(tasks))

# Dictionary zum Speichern der DataFrames pro Task & Typ
task_df_dict = {}

# JSON-Dateien einlesen
for task in tasks:
    for lin_type in lin_types:
        file_path = f"data/results_max_loss/intrinsic/ABv2_3ReLU_final_training{task}_{lin_type}.json"
        try:
            with open(file_path, "r") as f:
                df = pd.read_json(f, lines=True)
                df["task"] = task
                df["type"] = lin_type
                df["lr"] = df["lr"].astype(float)
                task_df_dict[(task, lin_type)] = df
        except FileNotFoundError:
            print(f"Fehlt: {file_path}")

# Plot
plt.figure(figsize=(10, 6))

for i, task in enumerate(tasks):
    for lin_type in lin_types:
        df = task_df_dict.get((task, lin_type))
        if df is None:
            continue

        # Mittelwert pro Lernrate
        summary_df = df.groupby("lr")["dist_inf"].mean().reset_index().sort_values("lr")

        linestyle = "-" if lin_type == "linear" else "--"
        label = f"{task} ({'Aff' if lin_type == 'linear' else 'nonlin'})"

        plt.plot(summary_df["lr"], summary_df["dist_inf"],
                 label=label,
                 color=colors[i],
                 linestyle=linestyle)

plt.xscale("log")
plt.xlabel("Learning rate")
plt.ylabel("Intrinsic distance")
plt.title("Intrinsic Homotopy Distances Across Tasks")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("avg_dist_per_lr.png", dpi=300)
plt.show()