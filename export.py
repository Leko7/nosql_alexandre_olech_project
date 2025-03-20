import matplotlib.pyplot as plt

def save_csv(df_sorted, csv_path):
    # Save as CSV
    df_sorted.to_csv(csv_path, index=False)

def save_png(df_sorted, png_path):

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(5, 2))  # Adjust size as needed
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=df_sorted.values, colLabels=df_sorted.columns, cellLoc='center', loc='center')

    # Save as PNG

    plt.savefig(png_path, bbox_inches="tight", dpi=300)