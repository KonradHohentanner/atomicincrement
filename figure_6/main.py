import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

def day_batch(ai: bool = False) -> int:

    path = (
        "./data/DAY_AI.ndjson"
    )
    data_day = (
        pl.scan_ndjson(path)
        .group_by(["batch_size", "date"])
        .agg([pl.mean("Top_1") * 100, pl.mean("Top_5") * 100, pl.mean("Top_10") * 100])
        .sort("date")
    )
    by_batches = data_day.collect().partition_by("batch_size")
    for batch in by_batches:
        if batch["batch_size"][0] == 256:
            basic_plots(batch, ai=ai)


    

def basic_plots(data: pl.DataFrame, ai: bool = False):
    x_labels = data["date"]
    x = list(range(0, len(x_labels)))
    t1, t5, t10 = data["Top_1"], data["Top_5"], data["Top_10"]
    t1 = gaussian_filter1d(t1, sigma=1)
    t5 = gaussian_filter1d(t5, sigma=1)
    t10 = gaussian_filter1d(t10, sigma=1)

    plt.rcParams.update(
        {
            "font.size": 24,  # General font size for all elements
            "axes.labelsize": 24,  # Axis label font size
            "axes.titlesize": 24,  # Title font size
            "xtick.labelsize": 24,  # X-axis tick label font size
            "ytick.labelsize": 24,  # Y-axis tick label font size
            "legend.fontsize": 20,  # Legend font size
        }
    )

    plt.ylim(-5, 105)
    plt.tick_params(axis="x", labelrotation=0)
    plt.plot(x, t1, marker="o", label="Top 1")
    plt.plot(x, t5, marker="o", label="Top 5")
    plt.plot(x, t10, marker="o", label="Top 10")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Days after first recorded trace")
    plt.tight_layout()
    if ai == False:
        plt.legend(loc="upper right")
    else:
        plt.legend(loc="lower right")
    plt.savefig(f"figure_6.pdf", format="pdf")
    plt.clf()


if __name__ == '__main__':
    day_batch(True)
