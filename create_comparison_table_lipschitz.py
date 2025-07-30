from collections import defaultdict
from matplotlib import font_manager

from xml.parsers.expat import model
import pandas as pd
from matplotlib.cm import ScalarMappable  # <-- FEHLTE!
from matplotlib.ticker import MaxNLocator
from pandas import concat
import matplotlib.pyplot as plt
import re
import seaborn as sns
import matplotlib

import json
import numpy as np
from matplotlib.patches import Patch
from scipy.spatial import ConvexHull
from scipy import stats
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import os

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




def get_values_from_json(task, intr_extr_type):
    if intr_extr_type != "extrinsic":
        intr_extr_type = ""
        lin_types = ["nonlinear", "linear"]
        df_list = []

        for lin in lin_types:
            file_path = f"/similarity_models/data/results/intrinsic/3ReLU_final_training{task}_{lin}.json"
            #file_path = f"training_results{intr_extr_type}_{task}_{lin}.json"
            with open(file_path, "r") as f:
                df_lin = pd.read_json(f, lines=True)
                df_list.append(df_lin)

        df = pd.concat(df_list, ignore_index=True)


    else:
        intr_extr_type = "_extrinsic"
        lin_type = ""
        with open(f"data/results/extrinsic/extrinsic_{task}_final_training.json", "r") as f:
            df = pd.read_json(f, lines=True)        
    # Kompakte Modellpaar-Labels
    def group_label(row):
        model1 = str(row["model_g"]).strip().lower()
        model2 = str(row["model_h"]).strip().lower()
        seed1 = int(row["seed_g"]) if pd.notna(row["seed_g"]) else ""
        seed2 = int(row["seed_h"]) if pd.notna(row["seed_h"]) else ""
        return f"{model1} {seed1} vs {model2} {seed2}"
    df["group"] = df.apply(group_label, axis=1)
    # Gruppen alphabetisch und numerisch sortieren
    sort_df = df[["model_g", "seed_g", "model_h", "seed_h", "group", "lr", "distance_inf_overall", "distance_std", "spearman", "pearson"]].copy()
    if intr_extr_type == "_extrinsic":
        # Extrinsic (Transformationen im extrinsischen Raum)
        sort_df[r"$L^{\mathcal{H},\,\mathrm{upper}}_{\mathcal{V}(V,\Delta),\,g}$"] = df["theoretical_lipschitz_g"].round(3)
        sort_df[r"$L^{\mathcal{H},\,\mathrm{lower}}_{\mathcal{V}(V,\Delta),\,g}$"] = df["empirical_bounds_g"].apply(
            lambda d: d["empirical_lower_bound"]
        ).round(3)
        sort_df[r"$L^{\mathcal{H},\,\mathrm{upper}}_{\mathcal{V}(V,\Delta),\,h}$"] = df["theoretical_lipschitz_h"].round(3)
        sort_df[r"$L^{\mathcal{H},\,\mathrm{lower}}_{\mathcal{V}(V,\Delta),\,h}$"] = df["empirical_bounds_h"].apply(
            lambda d: d["empirical_lower_bound"]
        ).round(3)
        sort_df[r"$\max(\mathrm{Loss})_{\mathcal{V}(V,\Delta)}$"] = df["train_loss_max"].round(3)
    else:
        # Intrinsic (z. B. lineare Maps im inneren Raum)
        sort_df[r"$L_{\mathrm{C}_g}^{\mathrm{upper}}$"] = df["lipschitz_upper"].round(3)
        sort_df[r"$L_{\mathrm{C}_g}^{\mathrm{lower}}$"] = df["lipschitz_lower"].round(3)
        sort_df[r"$\max(\mathrm{Loss})_{\mathrm{C}_g}$"] = df["train_loss_max"].round(3)
        if "epoch_losses" in df.columns:
            # Falls epoch_losses vorhanden ist, minimiere den Verlust
            sort_df["argmin(maxloss)"] = df["epoch_losses"].apply(lambda x: min(x[0]))
        sort_df["type"] = df["type"].apply(lambda x: "linear" if x == "linear" else "nonlinear")

    sort_df = sort_df.sort_values(by=["model_g", "seed_g", "model_h", "seed_h"])
    return sort_df


def format_sci(val):
    if pd.isna(val):
        return ""
    return f"{val:.1e}"
    # JSON-Datei laden
    
def model_sort_key(name):
    match = re.match(r"^(.*?)(?: \((\d+)\))?$", name)

    if match:
        model = match.group(1).strip().lower()
        seed_str = match.group(2)
        seed = int(seed_str) if seed_str is not None else -1  # fehlender Seed = -1
        return (model[0].upper(), seed)
    return (name[0].upper(), -1)

def model_sort_key_one_Letter(name):
    match = re.match(r"^(.*?)(?: \((\d+)\))?$", name)
    if match:
        model = match.group(1).strip()
        seed_str = match.group(2)
        seed = int(seed_str) if seed_str is not None else -1
        return (model, seed)
    return (name, -1)


def plot_lipschitz_constants(lipschitz_df, task, unique_groups, x_pos):
    print(f"[INFO] Plotting Lipschitz constants for task {task}..., that fulfill the theory")
    lrs = sorted(lipschitz_df["lr"].unique())
    lr_colors = plt.cm.Set2(np.linspace(0, 1, len(lrs)))
    lr_markers = ['o', 's', '^', 'D', 'v', '*']
    # Nur LRs mit mindestens einem Wert < 1.3 anzeigen
    filtered_lrs = sorted(
        lipschitz_df[lipschitz_df["lipschitz_upper"] < 1.3]["lr"].unique()
    )

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.set_title(f"Lipschitz-constant for task: {task}", fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    for lr_idx, lr in enumerate(filtered_lrs):
        lr_data = lipschitz_df[
            (lipschitz_df["lr"] == lr) & (lipschitz_df["lipschitz_upper"] < 1.3)
        ]
        if lr_data.empty:
            continue

        x = [x_pos[pair] for pair in lr_data["group"]]

        # Upper: durchgezogene Marker
        ax.scatter(
            x,
            lr_data["lipschitz_upper"],
            label=f"LR {lr} (upper)",
            color=lr_colors[lr_idx % len(lr_colors)],
            marker=lr_markers[lr_idx % len(lr_markers)],
            s=60,
            alpha=0.8
        )

        # Lower: leere Marker
        ax.scatter(
            x,
            lr_data["lipschitz_lower"],
            facecolors='none',
            edgecolors=lr_colors[lr_idx % len(lr_colors)],
            label=f"LR {lr} (lower)",
            marker=lr_markers[lr_idx % len(lr_markers)],
            s=60,
            alpha=0.8
        )

    # X-Achse: get groupnames on x Achses
    xticks = list(range(len(unique_groups)))
    xtick_labels = [label if i % 3 == 0 else "" for i, label in enumerate(unique_groups)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=9)

    ax.set_ylabel("Lipschitz-Konstanten (log)", fontsize=11)
    ax.legend(title=f"Learning Rates", loc='upper right')

    plt.tight_layout()
    plt.savefig(f"Filtered_lipschitz_constants_task_{task}.png", dpi=300, bbox_inches='tight')
    plt.close()


def intrinsic_lipschitz_constants(lipschitz_df, task):
    # 1. Filter: Nur Gruppen mit oberer Lipschitzkonstante < 1
    filtered_df = lipschitz_df[lipschitz_df[r"$L_{\mathrm{C}_g}^{\mathrm{upper}}$"] < 1.2].copy()
    if filtered_df.empty:
        print(f"[INFO] Keine Gruppen mit $L_{{C_g}}^{{upper}} < 1$ für Task '{task}' gefunden.")
        return

    # 2. Nach Lernraten gruppieren
    lrs = sorted(filtered_df["lr"].unique())
    n_cols = len(lrs)
    fig, axes = plt.subplots(nrows=1, ncols=n_cols, figsize=(25, 6), sharey=True)

    if n_cols == 1:
        axes = [axes]  # Für den Fall nur eines Subplots

    for ax, lr in zip(axes, lrs):
        df_lr = filtered_df[filtered_df["lr"] == lr].copy()
        #df_lr = df_lr.sort_values(by=r"$L_{\mathrm{C}_g}^{\mathrm{upper}}$")
        df_lr = df_lr.sort_values(by="group", key=lambda col: col.map(model_sort_key))


        # x-Achse = Gruppen
        x_pos = range(len(df_lr))
        group_labels = df_lr["group"]

        for i, row in enumerate(df_lr.itertuples()):
            lower = getattr(row, r"_9")  # je nach Spaltenposition ggf. anpassen
            upper = getattr(row, r"_8")

            ax.vlines(x=i, ymin=lower, ymax=upper, color="mediumblue", linewidth=2)
            ax.plot(i, lower, "o", color="skyblue", label="Lower bound" if i == 0 else "")
            ax.plot(i, upper, "o", color="tomato", label="Upper bound" if i == 0 else "")
            
        
        ax.set_title(f"lr = {lr}")

        # x-Achse: Gruppennamen, reduziert auf jedes 15. Label
        x_labels = list(df_lr["group"])
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(
            [label if i % 15 == 0 else '' for i, label in enumerate(x_labels)],
            rotation=90,
            fontsize=6
        )

        # y-Achse: Lipschitzkonstante
        ax.set_ylabel("Lipschitzkonstante")
        ax.set_ylim(0, 1.1)

        # Gitternetz und Stil
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Legende nur einmal anzeigen
        if ax == axes[0]:
            ax.legend(loc="upper right")


    plt.tight_layout()
    plt.savefig(f"data/results_max_loss/{task}_intrinsic_lipschitz_ranges.png", dpi=300)
    plt.close()




def point_plot_extr_vs_intr(dist_df, lipschitz_df, task):

    print(f"[INFO] Plotting intrinsic vs extrinsic distances for task {task}... where Lipschitzconstant fulfills the theory")

    dist_df["lr"] = dist_df["lr"].round(8)
    lipschitz_df["lr"] = lipschitz_df["lr"].round(8)

    # Schritt 1: Mergen wie zuvor
    merged_df = dist_df.merge(
        lipschitz_df[["group", "lr", r"$L_{\mathrm{C}_g}^{\mathrm{upper}}$", r"$L^{\mathcal{H},\,\mathrm{upper}}_{\mathcal{V}(V,\Delta),\,h}$", r"$L^{\mathcal{H},\,\mathrm{upper}}_{\mathcal{V}(V,\Delta),\,g}$"]],
        on=["group", "lr"],
        how="inner"
        #how="left"
    )
    # get lipschitz constants from lipschitz_df that are < 1.05
    filtered_df = merged_df[
        (merged_df[r"$L_{\mathrm{C}_g}^{\mathrm{upper}}$"] < 1.5) &
        (merged_df[r"$L^{\mathcal{H},\,\mathrm{upper}}_{\mathcal{V}(V,\Delta),\,h}$"] < 1.5) &
        (merged_df[r"$L^{\mathcal{H},\,\mathrm{upper}}_{\mathcal{V}(V,\Delta),\,g}$"] < 1.5)
    ]
    # cut of exploding values
    y_threshold = filtered_df[r"$d^\mathcal{H}_{\mathcal{V}(V,\Delta)}$"].quantile(0.99)  # oder: y_threshold = 1.0
    print(y_threshold)
    # get all distances that are below the threshold
    filtered_df = filtered_df[filtered_df[r"$d^\mathcal{H}_{\mathcal{V}(V,\Delta)}$"] <= y_threshold]
    # get LR
    lrs = sorted(filtered_df["lr"].unique())
    n = len(lrs[:4])

    # Layout definieren (z. B. 2 Spalten)
    cols = 2
    rows = int(np.ceil(n / cols))

    # Schritt 4: Subplots erstellen
    fig, axes = plt.subplots(1, n, figsize=(28, 6))


    axes = axes.flatten()
    fig.suptitle(f"Intrinsic vs Extrinsic Distances for Task: {task}", fontsize=24)

    for i, lr in enumerate(lrs[:4]):
        ax = axes[i]
        df_lr = filtered_df[filtered_df["lr"] == lr]

        x = df_lr[r"$d_{\mathrm{C}}$"]
        y = df_lr[r"$d^\mathcal{H}_{\mathcal{V}(V,\Delta)}$"]

        ax.scatter(x, y, c='tab:blue', edgecolor='black', label='Distance')
        ax.margins(x=0.1, y=0.1) 

        # Regressionsgerade berechnen
        if len(x) >= 2:  # nur wenn genug Punkte vorhanden
            coeffs = np.polyfit(x, y, 1)  # lineare Regression: y = mx + b
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = coeffs[0] * x_fit + coeffs[1]
            ax.plot(x_fit, y_fit, color="red", linestyle="--", label="Regression")
            ax.tick_params(axis='x', labelbottom=True, labelsize=14)
            ax.tick_params(axis='y', labelleft=True, labelsize=14)
        
        font_medium = font_manager.FontProperties(size=18)
        ax.set_title(f"lr = {lr}",fontproperties=font_medium)
        ax.set_xlabel(r"$d_{\mathrm{C}}$",fontproperties=font_medium)
        ax.set_ylabel(r"$d^\mathcal{H}_{\mathcal{V}(V,\Delta)}$",fontproperties=font_medium)

        ax.grid(True)
        ax.legend()

    # Leere Subplots deaktivieren
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"data/results_max_loss/subplots_intrinsic_vs_extrinsic_by_lr_{task}.png", dpi=300)
    plt.show()



def process_distance_as_heatmap(extrinsic_df, task, type):
    extrinsic_df = extrinsic_df.copy()

    extrinsic_df['seed_g'] = extrinsic_df['seed_g'].astype('Int64')
    extrinsic_df['seed_h'] = extrinsic_df['seed_h'].astype('Int64')

    extrinsic_df['G'] = np.where(
        extrinsic_df['seed_g'].notna(),
        extrinsic_df['model_g'] + " (" + extrinsic_df['seed_g'].astype(str) + ")",
        extrinsic_df['model_g']
    )
    extrinsic_df['H'] = np.where(
        extrinsic_df['seed_h'].notna(),
        extrinsic_df['model_h'] + " (" + extrinsic_df['seed_h'].astype(str) + ")",
        extrinsic_df['model_h']
    )

    extrinsic_df['G'] = extrinsic_df['G'].str.replace('<NA>', '', regex=False).str.strip()
    extrinsic_df['H'] = extrinsic_df['H'].str.replace('<NA>', '', regex=False).str.strip()

    extrinsic_df["lr"] = extrinsic_df["lr"].astype(float)
    unique_lrs = sorted(extrinsic_df["lr"].unique())
    n_lrs = len(unique_lrs)

    fig, axs = plt.subplots(1, n_lrs, figsize=(25, 6), squeeze=False)
    fig.suptitle(f"Heatmaps of {type} Distances for Task: {task}", fontsize=16)

    print(f"[INFO] Dataframes processed. Found {n_lrs} unique learning rates for task {task}")

    if type == "intrinsic":
        color = "RdPu"
        cbar_label = {'label': r'Intrinsic Distance $\|h - \psi(g)\|_\infty$'}
        model_x = "Target Model h"
        model_y = "Source Model g"
    else:
        color = "Reds"
        cbar_label = {'label': r'Extrinsic Distance $\|\psi(h) - \phi(g)\|_\infty$'}
        model_x = "Maximized Model h"
        model_y = "Minimized Model g"
    for i, lr in enumerate(unique_lrs):
        ax = axs[0][i]
        subset = extrinsic_df[extrinsic_df["lr"] == lr]

        heatmap_data = subset.pivot_table(
            index='G',
            columns='H',
            values='distance_inf_overall',
            aggfunc='mean'
        )

        all_models = sorted(set(heatmap_data.index) | set(heatmap_data.columns), key=model_sort_key)
        
        if "roberta" in all_models:
            all_models.remove("roberta")
            all_models = ["roberta"] + all_models
        heatmap_data = heatmap_data.reindex(index=all_models, columns=all_models)

        vmin = np.floor(heatmap_data.min().min() * 10) / 10
        vmax = np.ceil(heatmap_data.max().max() * 10) / 10


        hm = sns.heatmap(
            heatmap_data,
            cmap=color,
            annot=False,
            vmin=vmin,
            vmax=vmax,
            cbar=True,
            cbar_kws={
                'orientation': 'horizontal',
                'label': cbar_label['label'],
                'shrink': 0.7,          
                'pad': 0.30            
            },
            square=True,
            ax=ax
        )
        cbar = hm.collections[0].colorbar
        cbar.ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_title(f"lr = {lr}")
        ax.set_xlabel(model_x)
        ax.set_ylabel(model_y)

        # Vollständige x-Achse
        x_ticks = np.arange(len(all_models)) + 0.5
        x_labels = [label if i % 2 == 0 else '' for i, label in enumerate(all_models)]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=90)

        # Nur jedes zweite Label auf der y-Achse
        y_ticks = np.arange(len(all_models)) + 0.5
        y_labels = [label if i % 2 == 0 else '' for i, label in enumerate(all_models)]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, rotation=0)

        ax.invert_yaxis()
    # ⬅️ Position der gemeinsamen Colorbar (ggf. feintunen)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Platz lassen rechts für Colorbar
    plt.savefig(f"data/results_max_loss/Heatmap_{type}_distance_all_lrs_{task}.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[INFO] Heatmap for {type} distances processed and saved for task {task} with {n_lrs} unique learning rates.")


def process_lipschitz(intrinsic_df, extrinsic_df, task):
    print(f"[INFO] Processing Lipschitz constants for task {task}...")
    # Proces dataframes to get Lipschitz constants
    extrinsic_df[r"$d^\mathcal{H}_{\mathcal{V}(V,\Delta)}$"] = extrinsic_df["distance_inf_overall"].round(3)
    extrinsic_df[r"$\sigma(d^\mathcal{H}_{\mathcal{V}(V,\Delta)})$"] = extrinsic_df["distance_std"].round(3)
    extrinsic_df[r"Spearman$_{\mathcal{H}}$"] = extrinsic_df["spearman"].round(3)
    extrinsic_df[r"Pearson$_{\mathcal{H}}$"] = extrinsic_df["pearson"].round(3)
    columns_to_add = ["group", "lr"] + [
    r"$L_{\mathrm{C}_g}^{\mathrm{upper}}$",
    r"$L_{\mathrm{C}_g}^{\mathrm{lower}}$",
    r"$\max(\mathrm{Loss})_{\mathrm{C}_g}$"
    ]
    lipschitz_df = extrinsic_df.merge(
    intrinsic_df[intrinsic_df["type"] == "nonlinear"][columns_to_add],
    on=["group", "lr"],
    how="inner",
    suffixes=("", "_extr")
    )


    unique_groups = lipschitz_df["group"].unique().tolist()
    #dist_df = dist_df.drop(columns=["model_g", "seed_g", "model_h", "seed_h", "distance_inf_overall", "spearman", "pearson", "type"
    #])
    # remose unnecessary columns for lipschitz_df
    lipschitz_df= lipschitz_df.drop(columns=["model_g", "seed_g", "model_h", "seed_h", "distance_inf_overall", "distance_std", "spearman", "pearson", r"$d^\mathcal{H}_{\mathcal{V}(V,\Delta)}$",
    r"$\sigma(d^\mathcal{H}_{\mathcal{V}(V,\Delta)})$",
    r"Spearman$_{\mathcal{H}}$",
    r"Pearson$_{\mathcal{H}}$"])
    x_pos = {pair: i for i, pair in enumerate(lipschitz_df)}
    # Formatieren
    column_order = [
    "group", "lr",
    r"$L^{\mathcal{H},\,\mathrm{upper}}_{\mathcal{V}(V,\Delta),\,g}$",
    r"$L^{\mathcal{H},\,\mathrm{lower}}_{\mathcal{V}(V,\Delta),\,g}$",
    r"$L^{\mathcal{H},\,\mathrm{upper}}_{\mathcal{V}(V,\Delta),\,h}$",
    r"$L^{\mathcal{H},\,\mathrm{lower}}_{\mathcal{V}(V,\Delta),\,h}$",
    r"$\max(\mathrm{Loss})_{\mathcal{V}(V,\Delta)}$",
    r"$L_{\mathrm{C}_g}^{\mathrm{upper}}$",
    r"$L_{\mathrm{C}_g}^{\mathrm{lower}}$",
    r"$\max(\mathrm{Loss})_{\mathrm{C}_g}$"
    ]

    lipschitz_df = lipschitz_df[column_order]


    # Print Table as picture or if size is too large as latex table
    fig, ax = plt.subplots(figsize=(22, len(lipschitz_df) * 0.5 + 1))
    ax.axis('off')

    tbl = plt.table(cellText=lipschitz_df.values,
                    colLabels=lipschitz_df.columns,
                    loc='center',
                    cellLoc='center',
                    colLoc='center')

    tbl.auto_set_font_size(False)
    tbl.scale(1.1, 1.2)


    try:
        plt.savefig(f"data/results_max_loss/All_Lipschitzkonst_{task}.png", bbox_inches="tight", dpi=300)
        plt.close()
        if len(unique_groups) > 100:
                ## is size to tall, create latex table
            with open(f"data/results_max_loss/All_Lipschitzkonst_{task}.tex", "w") as f:
                f.write(lipschitz_df.to_latex(index=False))
    except Exception as e:
        print(f"[ERROR] error while saving: {e}")
        plt.close()

    intrinsic_lipschitz_constants(lipschitz_df, task )

    return lipschitz_df


def process_intrinsic_vs_extrinsic(intrinsic_df, extrinsic_df, task):
    print(f"[INFO] Processing intrinsic vs extrinsic distances for task {task}...")
    # Get required colums and rename them
    linear_df = intrinsic_df[intrinsic_df["type"] == "linear"].copy()
    nonlinear_df = intrinsic_df[intrinsic_df["type"] == "nonlinear"].copy()

    linear_df = linear_df[["group", "lr", "distance_inf_overall", "distance_std", "spearman", "pearson"]].rename(columns={
        "distance_inf_overall": r"$d_{\mathrm{Aff}}$",
        "distance_std": r"$\sigma(d_{\mathrm{Aff}})$",
        "spearman": r"Spearman$_{\mathrm{Aff}}$",
        "pearson": r"Pearson$_{\mathrm{Aff}}$"
    })

    nonlinear_df = nonlinear_df[["group", "lr", "distance_inf_overall", "distance_std", "spearman", "pearson"]].rename(columns={
        "distance_inf_overall": r"$d_{\mathrm{C}}$",
        "distance_std": r"$\sigma(d_{\mathrm{C}})$",
        "spearman": r"Spearman$_{\mathcal{C}}$",
        "pearson": r"Pearson$_{\mathcal{C}}$"
    })
    extrinsic_df = extrinsic_df[["group", "lr", "distance_inf_overall", "distance_std", "spearman", "pearson"]].rename(columns={
        "distance_inf_overall":  r"$d^\mathcal{H}_{\mathcal{V}(V,\Delta)}$",
        "distance_std": r"$\sigma(d^\mathcal{H}_{\mathcal{V}(V,\Delta)})$",
        "spearman": r"Spearman$_{\mathcal{H}}$",
        "pearson": r"Pearson$_{\mathcal{H}}$"
    })

    intrinsic_df = pd.merge(linear_df, nonlinear_df, on=["group", "lr"], how="outer")

    dist_columns_to_add = ["group", "lr"] + [
        r"$d^\mathcal{H}_{\mathcal{V}(V,\Delta)}$",
        r"$\sigma(d^\mathcal{H}_{\mathcal{V}(V,\Delta)})$",
        r"Spearman$_{\mathcal{H}}$",
        r"Pearson$_{\mathcal{H}}$"
    ]

    # get one df that contains all distances
    # Merge intrinsic and extrinsic distances

    dist_df = intrinsic_df.merge(
        extrinsic_df[dist_columns_to_add],
        on=["group", "lr"],
        how="inner"
    )

    # sort the columns
    column_order = [
        "group", "lr",
        r"$d_{\mathrm{Aff}}$", r"$\sigma(d_{\mathrm{Aff}})$", r"Spearman$_{\mathrm{Aff}}$", r"Pearson$_{\mathrm{Aff}}$",
        r"$d_{\mathrm{C}}$", r"$\sigma(d_{\mathrm{C}})$", r"Spearman$_{\mathcal{C}}$", r"Pearson$_{\mathcal{C}}$",
        r"$d^\mathcal{H}_{\mathcal{V}(V,\Delta)}$", r"$\sigma(d^\mathcal{H}_{\mathcal{V}(V,\Delta)})$",
        r"Spearman$_{\mathcal{H}}$", r"Pearson$_{\mathcal{H}}$"
    ]
    dist_df = dist_df[column_order]

    # Schritt 3: Formatieren
    def format_sci(val):
        if pd.isna(val):
            return ""
        return f"{val:.1e}"
    # format the columns
    for col in dist_df.columns:
        if "Spearman" in col or "Pearson" in col:
            dist_df[col] = dist_df[col].apply(format_sci)
        elif "$d_" in col or r"$\sigma(" in col:
            dist_df[col] = dist_df[col].round(3)

    # Show group names only in the first row
    dist_df_formatted = dist_df.sort_values(by=["group", "lr"]).reset_index(drop=True)
    dist_df_formatted["group"] = dist_df_formatted["group"].mask(dist_df_formatted["group"].duplicated(), "")
    #dist_df = dist_df[dist_df[r"$d_{\mathrm{C}}$"] < 1].copy()


    # save table graphically as png or latex table if size is too large
    fig, ax = plt.subplots(figsize=(22, len(dist_df) * 0.7))
    ax.axis('off')

    tbl = plt.table(
        cellText=dist_df_formatted.values,
        colLabels=dist_df_formatted.columns,
        loc='center',
        cellLoc='center',
        colLoc='center'
    )

    # Erste Spalte (group) breiter machen
    for key, cell in tbl.get_celld().items():
        if key[1] == 0:
            cell.set_width(0.2)

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.1, 1.3)

 
    try:
        plt.savefig(f"data/results_max_loss/Dist_extr_intr_lin_nonlin_{task}.png", bbox_inches="tight", dpi=300)
        plt.close()
        if len(dist_df) > 100:
                ## is size to tall, create latex table
            with open(f"data/results_max_loss/Dist_extr_intr_lin_nonlin_{task}.tex", "w") as f:
                f.write(dist_df.to_latex(index=False))
    except Exception as e:
        print(f"[ERROR] error while saving: {e}")
        plt.close()

    return dist_df

import pandas as pd
import matplotlib.pyplot as plt

def summarize_mean_distances_by_lr(dist_df, task, save=True):
    """
    Berechnet den Mittelwert der Distanzen pro Lernrate.

    Parameters:
    - dist_df: DataFrame mit Spalten wie $d_{\mathrm{Aff}}$, $d_{\mathrm{C}}$, $d^H_{V(V,Δ)}$ etc.
    - task: Task-Name für Dateibenennung
    - save: Wenn True, speichert PNG- und LaTeX-Datei

    Returns:
    - summary_df: DataFrame mit Mittelwerten pro Lernrate
    """

    # Wähle nur numerische Spalten für die Mittelwertberechnung
    #numeric_cols = [
    #    r"$d_{\mathrm{Aff}}$",
    #    r"$d_{\mathrm{C}}$",
    #    r"$d^\mathcal{H}_{\mathcal{V}(V,\Delta)}$"
    #]
    numeric_cols = [r"Spearman$_{\mathrm{Aff}}$",
        r"Pearson$_{\mathrm{Aff}}$",
        r"Spearman$_{\mathcal{C}}$",
        r"Pearson$_{\mathcal{C}}$",
        r"$d_{\mathrm{Aff}}$",
        r"$d_{\mathrm{C}}$"    ]
    # transform columns to numeric, errors='coerce' converts non-numeric values to NaN
    for col in numeric_cols:
        dist_df[col] = pd.to_numeric(dist_df[col], errors="coerce")
    summary_df = dist_df.groupby("lr")[numeric_cols].mean().reset_index()
    for col in summary_df.columns:
        if "Spearman" in col or "Pearson" in col or "lr" in col:
            summary_df[col] = summary_df[col].apply(format_sci)
        elif "$d_" in col or r"$\sigma(" in col:
            summary_df[col] = summary_df[col].round(5)

    if save:
        # Tabelle als PNG speichern
        fig, ax = plt.subplots(figsize=(10, len(summary_df) * 0.6))
        ax.axis('off')

        table = plt.table(
            cellText=summary_df.values,
            colLabels=summary_df.columns,
            loc='center',
            cellLoc='center',
            colLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.3)

        plt.savefig(f"data/results_max_loss/mean_distances_by_lr_{task}.png", bbox_inches="tight", dpi=300)
        plt.close()

        # Optional: LaTeX-Datei
        with open(f"data/results_max_loss/mean_distances_by_lr_{task}.tex", "w") as f:
            f.write(f"% Task: {task}\n")  # Kommentarzeile in LaTeX
            f.write(summary_df.to_latex(index=False))

    return summary_df



def summarize_mean_distances_by_lr_with_plot(dist_df, task, save=True):
    """
    Berechnet den Mittelwert der Distanzen pro Lernrate und erzeugt einen Plot.

    Parameters:
    - dist_df: DataFrame mit Spalten wie $d_{\mathrm{Aff}}$, $d_{\mathrm{C}}$, etc.
    - task: Task-Name für Dateibenennung
    - save: Wenn True, speichert PNG- und LaTeX-Dateien und den Plot

    Returns:
    - summary_df: DataFrame mit Mittelwerten pro Lernrate
    """

    # Definierte numerische Spalten
    numeric_cols = [
        r"Spearman$_{\mathrm{Aff}}$",
        r"Pearson$_{\mathrm{Aff}}$",
        r"Spearman$_{\mathcal{C}}$",
        r"Pearson$_{\mathcal{C}}$",
        r"$d_{\mathrm{Aff}}$",
        r"$d_{\mathrm{C}}$"
    ]

    # Umwandlung zu numerisch
    for col in numeric_cols:
        dist_df[col] = pd.to_numeric(dist_df[col], errors="coerce")

    # Gruppieren und mitteln
    summary_df = dist_df.groupby("lr")[numeric_cols].mean().reset_index()

    if save:
        # Tabelle als PNG speichern
        fig, ax = plt.subplots(figsize=(10, len(summary_df) * 0.6))
        ax.axis('off')
        table = plt.table(
            cellText=summary_df.round(4).values,
            colLabels=summary_df.columns,
            loc='center',
            cellLoc='center',
            colLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.3)
        plt.savefig(f"data/results_max_loss/mean_distances_by_lr_{task}.png", bbox_inches="tight", dpi=300)
        plt.close()

        # LaTeX export
        with open(f"data/results_max_loss/mean_distances_by_lr_{task}.tex", "w") as f:
            f.write(f"% Task: {task}\n")
            f.write(summary_df.to_latex(index=False))

        # Plot erstellen
        plt.figure(figsize=(7, 4))
        plt.plot(summary_df["lr"], summary_df[r"$d_{\mathrm{Aff}}$"], marker='o', label=r"$d_{\mathrm{Aff}}$")
        plt.plot(summary_df["lr"], summary_df[r"$d_{\mathrm{C}}$"], marker='s', label=r"$d_{\mathcal{C}}$")
        plt.xscale("log")
        plt.xlabel("Learning Rate")
        plt.ylabel("Average Distance")
        plt.title(f"Intrinsic Distances by Learning Rate ({task})")
        plt.legend()
        plt.grid(True, which="both", linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f"data/results_max_loss/plot_intrinsic_distances_{task}.png", dpi=300)
        plt.close()

    return summary_df



def get_latex_table(task, dist_df, lipschitz_df):

    # Schritt 1: Formatierung
    df = dist_df.copy()

    df = dist_df.merge(
        lipschitz_df[["group", "lr", r"$L_{\mathrm{C}_g}^{\mathrm{upper}}$", r"$L^{\mathcal{H},\,\mathrm{upper}}_{\mathcal{V}(V,\Delta),\,h}$", r"$L^{\mathcal{H},\,\mathrm{upper}}_{\mathcal{V}(V,\Delta),\,g}$"]],
        on=["group", "lr"],
        how="inner"
    )
    # Schritt 2: Gruppieren nach group
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        grouped[row["group"]].append(row)

    # Schritt 3: LaTeX-Tabellen-Code generieren
    latex_lines = []
    latex_lines.append(r"\begin{tabular}{|c|c|" + "c|" * (len(df.columns) - 2) + "}")
    latex_lines.append(r"\hline")
    latex_lines.append(" & ".join(df.columns) + r" \\")
    latex_lines.append(r"\hline")

    for group, rows in grouped.items():
        rowspan = len(rows)
        for i, row in enumerate(rows):
            row_values = [str(row[col]) for col in df.columns]
            if i == 0:
                row_values[0] = rf"\multirow{{{rowspan}}}{{*}}{{{group}}}"
            else:
                row_values[0] = ""
            latex_lines.append(" & ".join(row_values) + r" \\")
        latex_lines.append(r"\hline")

    latex_lines.append(r"\end{tabular}")
    latex_code = "\n".join(latex_lines)

    # Speichern in Datei
    with open(f"data/final_latex_table/vergleichstabelle_mit_multirow_{task}.tex", "w") as f:
        f.write(latex_code)

    return latex_code

def plot_all_intrinsic_distances(task_df_dict):
    """
    Plottet einen kombinierten Linienplot für alle Tasks in einer einzigen Abbildung.
    Für jeden Task wird eine Farbe gewählt, die für Affin (solid) und Nichtlinear (dashed) verwendet wird.
    """
    plt.figure(figsize=(10, 6))

    # Farbpalette für Tasks
    colors = sns.color_palette("tab10", n_colors=len(task_df_dict))
    
    for i, (task, df) in enumerate(task_df_dict.items()):
        summary_df = df.groupby("lr")[["$d_{\mathrm{Aff}}$", "$d_{\mathrm{C}}$"]].mean().reset_index()

        # Sortieren der Lernraten für konsistente Linien
        summary_df = summary_df.sort_values("lr")

        # Affine Transformation: durchgezogen
        plt.plot(summary_df["lr"], summary_df["$d_{\mathrm{Aff}}$"], 
                 label=f"{task} (Aff)", color=colors[i], linestyle="-")
        
        # Nichtlineare Transformation: gestrichelt
        plt.plot(summary_df["lr"], summary_df["$d_{\mathrm{C}}$"], 
                 label=f"{task} (nonlin)", color=colors[i], linestyle="--")

    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("Intrinsic distance")
    plt.title("Intrinsic Homotopy Distances Across Tasks")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(False)
    plt.savefig("data/results_max_loss/intrinsic_distance_all_tasks.png", dpi=300)
    plt.show()

def plot_all_intrinsic_distances(task_df_dict):
    """
    Plottet einen kombinierten Linienplot für alle Tasks in einer einzigen Abbildung.
    Für jeden Task wird eine Farbe gewählt, die für Affin (solid) und Nichtlinear (dashed) verwendet wird.
    """
    plt.figure(figsize=(10, 6))

    # Farbpalette für Tasks
    colors = sns.color_palette("tab10", n_colors=len(task_df_dict))
    
    for i, (task, df) in enumerate(task_df_dict.items()):
        summary_df = df.groupby("lr")[["$d_{\mathrm{Aff}}$", "$d_{\mathrm{C}}$"]].mean().reset_index()

        # Sortieren der Lernraten für konsistente Linien
        summary_df = summary_df.sort_values("lr")

        # Affine Transformation: durchgezogen
        plt.plot(summary_df["lr"], summary_df["$d_{\mathrm{Aff}}$"], 
                 label=f"{task} (Aff)", color=colors[i], linestyle="-")
        
        # Nichtlineare Transformation: gestrichelt
        plt.plot(summary_df["lr"], summary_df["$d_{\mathrm{C}}$"], 
                 label=f"{task} (nonlin)", color=colors[i], linestyle="--")

    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("Intrinsic distance")
    plt.title("Intrinsic Homotopy Distances Across Tasks")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(False)
    plt.savefig("intrinsic_distance_all_tasksPräsi.png", dpi=300)
    plt.show()

def get_homotopic_groups(df, lin_type, task):

    """
    Findet Paare (g,h) mit symmetrischer Nähe in beide Richtungen.
    
    Args:
        dist_df: DataFrame mit Spalten model_g, model_h, seed_g, seed_h, distance_col
        distance_col: Spaltenname der Distanzmetrik
        tol: Toleranzschwelle, unterhalb derer ein Paar als 'nah' gilt

    Returns:
        DataFrame mit allen (g,h), die sich wechselseitig annähern (d < tol in beide Richtungen)
    """
    # Create canonical group keys for easy matching
    if lin_type == "intrinsic":
        dist_df = df[df['$L_{\\mathrm{C}_g}^{\\mathrm{upper}}$'] < 1.5].copy()
    if lin_type == "extrinsic":
        dist_df = df[df['$L^{\\mathcal{H},\\,\\mathrm{upper}}_{\\mathcal{V}(V,\\Delta),\\,h}$'] < 1.5].copy()
        dist_df = dist_df[dist_df['$L^{\\mathcal{H},\\,\\mathrm{upper}}_{\\mathcal{V}(V,\\Delta),\\,g}$'] < 1.5].copy()
        dist_df["type"] = "extrinsic"
    distance_col="distance_inf_overall"
    tol=4
    dist_df["group_key"] = dist_df.apply(
        lambda row: f"{row['lr']}|{row['type']}|{row['model_g']}-{row['seed_g']}|{row['model_h']}-{row['seed_h']}", axis=1
    )
    dist_df["reverse_key"] = dist_df.apply(
        lambda row: f"{row['lr']}|{row['type']}|{row['model_h']}-{row['seed_h']}|{row['model_g']}-{row['seed_g']}", axis=1
    )

    # Erstelle ein Lookup für schnelle Abfrage der Rückrichtung
    dist_lookup = dist_df.set_index("group_key")[distance_col].to_dict()

    results = []
    for idx, row in dist_df.iterrows():
        direct_dist = row[distance_col]
        reverse_dist = dist_lookup.get(row["reverse_key"], None)

        if reverse_dist is not None and direct_dist <= tol and reverse_dist <= tol:
            results.append({
                "model_g": row["model_g"],
                "seed_g": row["seed_g"],
                "model_h": row["model_h"],
                "seed_h": row["seed_h"],
                "lr": row["lr"],
                "dist_g_psi_h": direct_dist,
                "dist_h_phi_g": reverse_dist,
                "type": row["type"]
            })

    return pd.DataFrame(results)

def process_distance_as_heatmap_with_homotopy(extrinsic_df, task, lin_type, homotopic_pairs):
    name_map = {
    "roberta": "R",
    "multiberts": "M",
    "electra": "E",
}
    
    extrinsic_df['model_g'] = extrinsic_df['model_g'].map(name_map).fillna(extrinsic_df['model_g'])
    extrinsic_df['model_h'] = extrinsic_df['model_h'].map(name_map).fillna(extrinsic_df['model_h'])
    extrinsic_df = extrinsic_df.copy()
    extrinsic_df['seed_g'] = extrinsic_df['seed_g'].astype('Int64')
    extrinsic_df['seed_h'] = extrinsic_df['seed_h'].astype('Int64')

    extrinsic_df['G'] = np.where(
        extrinsic_df['seed_g'].notna(),
        extrinsic_df['model_g'] + " (" + extrinsic_df['seed_g'].astype(str) + ")",
        extrinsic_df['model_g']
    )
    extrinsic_df['H'] = np.where(
        extrinsic_df['seed_h'].notna(),
        extrinsic_df['model_h'] + " (" + extrinsic_df['seed_h'].astype(str) + ")",
        extrinsic_df['model_h']
    )

    extrinsic_df['G'] = extrinsic_df['G'].str.replace('<NA>', '', regex=False).str.strip()
    extrinsic_df['H'] = extrinsic_df['H'].str.replace('<NA>', '', regex=False).str.strip()

    extrinsic_df["lr"] = extrinsic_df["lr"].astype(float)
    unique_lrs = sorted(extrinsic_df["lr"].unique())
    n_lrs = len(unique_lrs)


    if lin_type == "linear" or lin_type == "nonlinear":
        
        model_x = "Target Model h"
        model_y = "Source Model g"
        if lin_type == "linear":
            extrinsic_df = extrinsic_df[extrinsic_df["type"] == "linear"]  
            color = "OrRd"
            cbar_label = {'label': r'Intrinsic linear Distance $\|h - \psi(g)\|_\infty$'}
        if lin_type == "nonlinear":
            extrinsic_df = extrinsic_df[extrinsic_df["type"] == "nonlinear"]
            color = "RdPu"
            cbar_label = {'label': r'Intrinsic nonlinear Distance $\|h - \psi(g)\|_\infty$'}
    else:
        color = "Reds"
        cbar_label = {'label': r'Extrinsic Distance $\|\psi(h) - \phi(g)\|_\infty$'}
        model_x = "Maximized Model h"
        model_y = "Minimized Model g"
        extrinsic_df["type"] = "extrinsic"


    #n_cols = math.ceil(len(unique_lrs[:4]) / 2)
    n_cols = 4
    n_rows = 1
    fig, axs = plt.subplots(1, 4, figsize=(4.1 * n_cols, 5*n_rows), squeeze=False)


    fig.suptitle(f"Heatmaps of {lin_type} Distances for Task: {task}", fontsize=16)

    for i, lr in enumerate(unique_lrs[:4]):
        row = i // n_cols
        col = i % n_cols
        ax = axs[row][col]

        subset = extrinsic_df[extrinsic_df["lr"] == lr]

        heatmap_data = subset.pivot_table(
            index='G',
            columns='H',
            values='distance_inf_overall',
            aggfunc='mean'
        )

        all_models = sorted(set(heatmap_data.index) | set(heatmap_data.columns), key=model_sort_key_one_Letter)
        #all_models = [model_sort_key_one_Letter(name) for name in sorted(set(heatmap_data.index) | set(heatmap_data.columns))]

        if "R" in all_models:
            all_models.remove("R")
            all_models = ["R"] + all_models
        heatmap_data = heatmap_data.reindex(index=all_models, columns=all_models)

        vmin = np.floor(heatmap_data.min().min() * 10) / 10
        vmax = np.ceil(heatmap_data.max().max() * 10) / 10

        hm = sns.heatmap(
            heatmap_data,
            cmap=color,
            annot=False,
            vmin=vmin,
            vmax=vmax,
            cbar=True,
            cbar_kws={
                'orientation': 'horizontal',
                'label': cbar_label['label'],
                'shrink': 0.8,
                'pad': 0.20
            },
            square=True,
            ax=ax
        )
        cbar = hm.collections[0].colorbar
        cbar.ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_title(f"lr = {lr}")
        ax.set_xlabel(model_x)
        ax.set_ylabel(model_y)

        always_show = {"R", "E"}

        x_ticks = np.arange(len(all_models)) + 0.5
        x_labels = [label if (label in always_show or i % 2 == 0) else "" for i, label in enumerate(all_models)]

        ax.set_xticks(x_ticks)
        font_small = font_manager.FontProperties(size=7)

        ax.set_xticklabels(x_labels, rotation=90,fontproperties=font_small)

        y_ticks = np.arange(len(all_models)) + 0.5

        y_labels = [label if (label in always_show or i % 2 == 0) else "" for i, label in enumerate(all_models)]
        #y_labels = [label if i % 2 == 0 else '' for i, label in enumerate(all_models)]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, rotation=0,fontproperties=font_small)

        ax.invert_yaxis()

        def get_model_label(model, seed):
            return f"{model[0].upper()} ({int(seed)})" if not pd.isna(seed) else model[0].upper()

        #for _, row_hom in homotopic_pairs[homotopic_pairs["type"] == lin_type].iterrows():
        #    if row_hom['lr'] == lr:
        #        i_label = get_model_label(row_hom["model_g"], row_hom["seed_g"])
        #        j_label = get_model_label(row_hom["model_h"], row_hom["seed_h"])
        #        if i_label in all_models and j_label in all_models:
        #            i = all_models.index(i_label)
        #            j = all_models.index(j_label)
        #            ax.plot(j + 0.5, i + 0.5, marker='o', markersize=3, color='white',
        #                    markeredgecolor='black', linewidth=0.3)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    #plt.subplots_adjust(wspace=0.01)  # oder 0.01 für sehr engen Abstand


    plt.savefig(f"PräsiHeatmap_{lin_type}_distance_all_lrs_{task}_homotopy.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_homotopic_heatmap(homotopic_df, task_name="All Tasks", save_path="homotopic_heatmap.png"):
    """
    Visualisiert homotope Modellpaare in einer Heatmap.
    """
    # Kombiniere Modelle mit Seed als Identifier
    homotopic_df["model_g_full"] = homotopic_df["model_g"].astype(str) + "(" + homotopic_df["seed_g"].astype(str) + ")"
    homotopic_df["model_h_full"] = homotopic_df["model_h"].astype(str) + "(" + homotopic_df["seed_h"].astype(str) + ")"

    # Erstelle eine Pivot-Tabelle
    pivot_df = pd.pivot_table(
        homotopic_df,
        values="dist_g_psi_h",  # oder np.mean aus beiden Richtungen
        index="model_g_full",
        columns="model_h_full"
    )

    # Symmetrisieren (optional)
    pivot_df = pivot_df.combine_first(pivot_df.T)

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="viridis", linewidths=0.5, square=True)
    plt.title(f"Intrinsic Homotopy Distance Matrix – {task_name}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_summarized_dfs(summarized_dfs):
    """
    Speichert die zusammengefassten DataFrames in einer CSV-Datei.
    
    Args:
        summarized_dfs: Dictionary mit Tasknamen als Schlüss
    """
    for task, df in summarized_dfs.items():
        df.to_csv(f"data/results_max_loss/summary_mean_distances_by_lr_{task}.csv", index=False)
    
    # DataFrames mit zusätzlicher 'Task'-Spalte versehen
    for task, df in summarized_dfs.items():
        df["Task"] = task
        summarized_dfs[task] = df

    # Alle DataFrames zu einem einzigen zusammenführen
    combined_df = concat(summarized_dfs.values(), ignore_index=True)

    # Task-Spalte als erste Spalte anordnen
    cols = ["Task"] + [col for col in combined_df.columns if col != "Task"]
    combined_df = combined_df[cols]
    plot_homotopic_heatmap(combined_df, task_name="All Tasks", save_path="homotopic_heatmap.png")

    # In LaTeX-Datei schreiben
    with open("data/results_max_loss/summary_mean_distances_<1.5_by_lr.tex", "w") as f:
        f.write("% Summary of mean distances by learning rate for all tasks\n")
        f.write(combined_df.to_latex(index=False))
    return combined_df


def merge_for_comparison(intrinsic_df, extrinsic_df):

    # Repräsentation vereinheitlichen
    def model_label(row):
        return f"{row['model_g']}({int(row['seed_g'])}) → {row['model_h']}({int(row['seed_h'])})" \
            if not pd.isna(row["seed_g"]) and not pd.isna(row["seed_h"]) else f"{row['model_g']} → {row['model_h']}"

    intrinsic_df["pair"] = intrinsic_df.apply(model_label, axis=1)
    extrinsic_df["pair"] = extrinsic_df.apply(model_label, axis=1)

    # für jeden Pair+lr das Minimum aus beiden Richtungen nehmen
    intrinsic_df["intrinsic_dist"] = intrinsic_df[["dist_g_psi_h"]]
    extrinsic_df["extrinsic_dist"] = extrinsic_df[["dist_g_psi_h"]]

    # Auswahl: nur gemeinsam vorkommende Paare und gleiche Lernraten
    merged = pd.merge(
        intrinsic_df[["pair", "lr", "intrinsic_dist"]],
        extrinsic_df[["pair", "lr", "extrinsic_dist"]],
        on=["pair", "lr"],
        how="inner"
    )

    return merged

def plot_train_loss_(intrinsic_unique_groups, extrinsic_unique_groups, task):
    os.makedirs("data/final_plots", exist_ok=True)

    # Nur die ersten 4 Lernraten verwenden
    top4_lrs = sorted(intrinsic_unique_groups['lr'].unique())
    df = intrinsic_unique_groups[intrinsic_unique_groups['lr'].isin(top4_lrs)].copy()
    df_ex = extrinsic_unique_groups[extrinsic_unique_groups['lr'].isin(top4_lrs)].copy()
    # Zielmodellbezeichner formatieren
    def format_label(row):
        model = row['model_h'].lower()
        if model.startswith('electra'):
            return 'E'
        elif model.startswith('roberta'):
            return 'R'
        elif model.startswith('multiberts'):
            return f"M{int(row['seed_h'])}"
        else:
            return row['model_h']
    df['target_label'] = df.apply(format_label, axis=1)
    df_ex['target_label'] = df_ex.apply(format_label, axis=1)
    # Gruppieren: Median + Quartile für max(Loss)
    grouped = df.groupby(['type', 'target_label'])['$\max(\mathrm{Loss})_{\mathrm{C}_g}$'].agg(
        median='median',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    ).reset_index()
    #grouped_ex = df_ex.groupby('target_label')['argmin(maxloss)'].agg(
    #    median='median',
    #    q25=lambda x: x.quantile(0.25),
    #    q75=lambda x: x.quantile(0.75)
    #).reset_index()
    # Sortierschlüssel
    def sort_key(label):
        if label == 'E': return (0, 0)
        if label == 'R': return (1, 0)
        if label.startswith('M'): return (2, int(label[1:]))
        return (3, label)

    grouped['label_sort'] = grouped['target_label'].apply(sort_key)
    #grouped_ex['label_sort'] = grouped_ex['target_label'].apply(sort_key)
    grouped = grouped.sort_values(by='label_sort')
    #grouped_ex = grouped_ex.sort_values(by='label_sort')

    # Plot vorbereiten
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(f"Max Training Loss for {task.upper()}", fontsize=16)

    for ax, t in zip(axes, ['linear', 'nonlinear']):
        data = grouped[grouped['type'] == t]
        y = data['median']
        x =  np.arange(len(data))
        #yerr = [y - data['q25'], data['q75'] - y]

        #ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=4, color='tab:blue', ecolor='gray')
        ax.scatter(x, y, marker='o', linestyle='-', color='tab:blue', label=f"{t.capitalize()} Median")

        ax.set_xticks(x)
        ax.set_xticklabels(data['target_label'], rotation=90)
        ax.set_title(f"{t.capitalize()} Mappings")
        ax.set_xlabel("Target Encoder")
        ax.set_ylabel(r'$\max(\mathrm{Loss})_{\mathrm{C}_g}$')

    plt.tight_layout()
    plt.savefig(f"data/results_max_loss/train_loss_intrinsic_extrinsic_median{task}.png", bbox_inches="tight", dpi=300)


def plot_train_loss(intrinsic_unique_groups, task):
    os.makedirs("data/final_plots", exist_ok=True)

    # Nur die letzten 4 Lernraten verwenden
    top4_lrs = sorted(intrinsic_unique_groups['lr'].unique())[2]
    df = intrinsic_unique_groups[intrinsic_unique_groups['lr'].isin([top4_lrs])].copy()

    # Zielmodellbezeichner formatieren
    def format_label(row):
        model = row['model_h'].lower()
        if model.startswith('electra'):
            return 'E'
        elif model.startswith('roberta'):
            return 'R'
        elif model.startswith('multiberts'):
            return f"M{int(row['seed_h'])}"
        else:
            return row['model_h']
    df['target_label'] = df.apply(format_label, axis=1)

    # Gruppieren: Median + Quartile für 
#    grouped = df.groupby(['type', 'target_label'])['$\max(\mathrm{Loss})_{\mathrm{C}_g}$'].agg(
    grouped = df.groupby(['type', 'target_label'])["argmin(maxloss)"].agg(
        mean='mean',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    ).reset_index()

    # Sortierschlüssel
    def sort_key(label):
        if label == 'E': return (0, 0)
        if label == 'R': return (1, 0)
        if label.startswith('M'): return (2, int(label[1:]))
        return (3, label)

    grouped['label_sort'] = grouped['target_label'].apply(sort_key)
    grouped = grouped.sort_values(by='label_sort')

    # Plot vorbereiten
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(f"Max Training Loss for {task.upper()}", fontsize=16)

    for ax, t in zip(axes, ['linear', 'nonlinear']):
        data = grouped[grouped['type'] == t]
        x = np.arange(len(data))
        y = data['mean']
        yerr = [y - data['q25'], data['q75'] - y]

        #ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=4, color='tab:blue', ecolor='gray')
        ax.scatter(x, y, marker='o', linestyle='-', color='tab:blue', label=f"{t.capitalize()} Mean")
        ax.set_xticks(x)
        ax.set_xticklabels(data['target_label'], rotation=90)
        ax.set_title(f"{t.capitalize()} Mappings")
        ax.set_xlabel("Target Encoder")
        ax.set_ylabel(r'$\max(\mathrm{Loss})_{\mathrm{C}_g}$')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"data/results_max_loss/train_loss_{task}_scatter.png", dpi=300, bbox_inches="tight")


def plot_intrinsic_vs_extrinsic_subplots(merged_df, task, title="Intrinsic vs Extrinsic Homotopy"):
    # Lernraten sortieren und konvertieren
    merged_df["lr"] = merged_df["lr"].astype(str)
    lrs = sorted(merged_df["lr"].unique(), key=lambda x: float(x))
    n_lrs = len(lrs)

    # Subplots vorbereiten
    fig, axes = plt.subplots(1, n_lrs, figsize=(5 * n_lrs, 5), sharey=True)

    for i, lr_val in enumerate(lrs):
        ax = axes[i] if n_lrs > 1 else axes
        subset = merged_df[merged_df["lr"] == lr_val]

        # Scatterplot
        sns.scatterplot(
            data=subset,
            x="intrinsic_dist",
            y="extrinsic_dist",
            ax=ax,
            color="navy",
            edgecolor="white",
            s=30
        )

        # Regression
        sns.regplot(
            data=subset,
            x="intrinsic_dist",
            y="extrinsic_dist",
            scatter=False,
            ax=ax,
            line_kws={"color": "black"},
            ci=None,
        )

        # Schwellenlinien
        ax.axhline(1, color='red', linestyle='--', linewidth=0.5)
        ax.axvline(1, color='red', linestyle='--', linewidth=0.5)

        # Achsentitel und Subtitels
        ax.set_title(f"lr = {lr_val}")
        ax.set_xlabel(r"$d_{\mathrm{C}(h,g)}$")
        if i == 0:
            ax.set_ylabel(r"$d^\mathcal{H}_{\mathcal{V}(V,\Delta)}(h,g)$")
        else:
            ax.set_ylabel("")

        ax.grid(True)

    fig.suptitle(f"{title}\n(Task: {task})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"data/results_max_loss/Dist_extr_intr_subplots_{task}.png", bbox_inches="tight", dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract embeddings and labels from a model.")
    parser.add_argument("--root_path", type=str, default="data/nonlin/", help="Path to the directory where the extracted embeddings are stored.")
    parser.add_argument("--task_name", type=str, default=["cola", "mrpc", "qnli", "qqp", "sst2","rte"], nargs="+", help="Name(s) of the task(s) (e.g., sst2, mnli).")
    parser.add_argument("--num_seeds", type=int, default=25, help="Number of seeds to use for generating embeddings.")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of labels for the classification task.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached preprocessed datasets or not.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory where the embeddings and labels will be saved.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for the dataloaders.")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs for training.")
    parser.add_argument("--layer", type=int, default=12, help="Layer of the model to extract embeddings from.")
    parser.add_argument("--hidden_size", type=int, default=128, help="Size of the hidden layer in the model.")
    parser.add_argument("--learning_rates", type=float, nargs="+", default=[0.00001, 0.0001, 0.001], help="List of learning rates to use for training.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for computation (e.g., cpu or cuda).")
    parser.add_argument("--model_name_path", default="roberta-base", type=str, help="Path to the pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--model_type", type=str, default=["electra", "roberta", "multiberts"],  help="Type of model to use (bert, roberta, electra).")
    parser.add_argument("--cache_dir", type=str, default="data/cache", help="Directory to store the pretrained models downloaded from huggingface.co.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per device for evaluation.")
    parser.add_argument("--activation", type=str, default="3Relu_multiProcess", help="Name of the activation function used in the model (e.g., _3_ReLu, _4_Sigmoid).")

    args = parser.parse_args()
    tasks = args.task_name
    intr_extr_tytes = [ "extrinsic", "intrinsic"]
    all_dist_dfs = {}

    summarized_dfs = {}

    for task in tasks:
        # Get intrinsic and extrinsic data on task
        intrinsic_unique_groups = get_values_from_json(task, "intrinsic")
        
        print(f"[INFO] Found {len(intrinsic_unique_groups)} intrinsic groups for task {task}")
        extrinsic_unique_groups = get_values_from_json(task, "extrinsic")
        #plot_train_loss(intrinsic_unique_groups, task)
        #print(f"[INFO] Found {len(extrinsic_unique_groups)} extrinsic groups for task {task}")
        #extrinsic_homotopic_groups = get_homotopic_groups(extrinsic_unique_groups, 'extrinsic', task)

        intrinsic_homotopic_groups = get_homotopic_groups(intrinsic_unique_groups,'intrinsic', task)
        #extrinsic_homotopic_groups = get_homotopic_groups(extrinsic_unique_groups,'extrinsic', task)
        # Create heatmaps for each LR to get the distance for each modelpair and task
        #nonlin_intrinsic_df = intrinsic_unique_groups[intrinsic_unique_groups["type"] == "nonlinear"]
        #process_distance_as_heatmap_with_homotopy(extrinsic_unique_groups, task, "extrinsic", [])
        process_distance_as_heatmap_with_homotopy(intrinsic_unique_groups, task, "linear", intrinsic_homotopic_groups)
        process_distance_as_heatmap_with_homotopy(intrinsic_unique_groups, task, "nonlinear", intrinsic_homotopic_groups)

        #process_distance_as_heatmap(extrinsic_unique_groups, task, "extrinsic")
        # Create a table that contains all Lipschitz constants and trainingloss 
        #lipschitz_df = process_lipschitz(intrinsic_unique_groups, extrinsic_unique_groups, task)
        # Create a table that contains the distances for each modelpair and task
        #summarized_df = summarize_mean_distances_by_lr(lipschitz_df, task)
        #merged_df = merge_for_comparison( intrinsic_homotopic_groups[intrinsic_homotopic_groups["type"] == "nonlinear"], extrinsic_homotopic_groups)
        #plot_intrinsic_vs_extrinsic_subplots(merged_df, task, title=f"Intrinsic vs Extrinsic Homotopy Distances for Task: {task}")
        dist_df = process_intrinsic_vs_extrinsic(intrinsic_unique_groups, extrinsic_unique_groups, task)
        all_dist_dfs[task] = dist_df

        #summarized_df = summarize_mean_distances_by_lr_with_plot(dist_df, task)
        #summarized_dfs[task] = intrinsic_homotopic_groups

        # Plot all lipschitz constants that fulfill the theory
        # plot_lipschitz_constants(lipschitz_df, task, unique_groups, x_pos)

        # Plot all distances that fulfill the theory (Lipschitz <=1)
        #point_plot_extr_vs_intr(dist_df, lipschitz_df, task)
        #print(f"[INFO] Processed intrinsic and extrinsic distances for task {task}.")
        #get_latex_table(task, dist_df, lipschitz_df)

    #plot_all_intrinsic_distances(all_dist_dfs)
    #save_summarized_dfs(summarized_dfs)
