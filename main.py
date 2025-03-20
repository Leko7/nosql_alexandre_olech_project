import torch

from DatasetsList import get_datasets_list
from normalization import *
from distance import *
from evaluation import evaluation
from ResultsTable import get_results_table
from export import save_csv, save_png

distance_metrics = {
    "ED": lambda x, y: torch.cdist(x, y, p=2),
    "Lorentzian": lorentzian,
    "Manhattan": manhattan,
    "Avg_l1_linf": avg_l1_linf,
    "Jaccard": jaccard,
    "Minkowski": lambda x, y: minkowski(x, y, p=3)  # Default p=3
}

normalizations = {
    "z-score" : z_score_norm,
    "MinMax" : min_max_norm,
    "UnitLength" : unit_length_norm,
    "MeanNorm" : mean_norm,
    "Tanh" : tanh_norm
}

root_dir = 'UCRArchive_2018'
threshold = 1e8 # 80 datasets for 1e8
datasets_list = get_datasets_list(threshold, root_dir)
batch_size = 256
results = []

for i, selected_norm in enumerate(list(normalizations.keys())):
    normalization = normalizations[selected_norm]

    for j, selected_dist in enumerate(list(distance_metrics.keys())):
        distance = distance_metrics[selected_dist]
        print(f"Evaluation {i*5 + j}, Distance : {selected_dist}, Normalization : {selected_norm}")
        results.append(evaluation(selected_dist, selected_norm, datasets_list, root_dir, batch_size, distance_metrics, normalizations))

df_sorted = get_results_table(results)

png_path = "table_2_1e8_full.png"
save_png(df_sorted, png_path)

csv_path = "table_2_1e8_full_sorted.csv"
save_csv(df_sorted, csv_path)