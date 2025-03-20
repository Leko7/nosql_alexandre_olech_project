import pandas as pd

def get_results_table(results):

    results_df = pd.DataFrame(results)

    sorting_order = {
        "Distance Measure": ["Minkowski", "Lorentzian", "Manhattan", "Avg_l1_linf", "DISSIM", "Jaccard", "ED"],
        "Scaling Method": ["z-score", "MinMax", "UnitLength", "MeanNorm", "Tanh"]
    }

    results_df["Distance Measure"] = pd.Categorical(results_df["Distance Measure"], categories=sorting_order["Distance Measure"], ordered=True)
    results_df["Scaling Method"] = pd.Categorical(results_df["Scaling Method"], categories=sorting_order["Scaling Method"], ordered=True)

    # Sorting based on the new categorical ordering
    df_sorted = results_df.sort_values(by=["Distance Measure", "Scaling Method"]).reset_index(drop=True)

    # Round like in the article
    df_sorted['Average Accuracy'] = round(df_sorted['Average Accuracy'],4)

    return df_sorted