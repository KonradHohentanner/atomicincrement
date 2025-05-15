import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d


def condition(ai: bool = False) -> int:
    def opposite(basic_acc, basic_count, acc, count):
        opposite_count = basic_count - count
        opposite_acc = (basic_acc * basic_count - acc * count) / opposite_count
        return opposite_acc, opposite_count

    path = (
        "./data/DEFAULT_AI.ndjson"
    )

    dataset_acc = pl.scan_ndjson(path).collect()
    df = dataset_acc.to_pandas()
    df["condition"] = df["condition"].map(lambda x: "idle" if x == "lightidle" else x)
    df["condition"] = df["condition"].map(lambda x: "ps" if x == "powersafe" else x)

    opp_conditions_for = ["b_20", "ram_115"]
    for c in opp_conditions_for:
        for device in df["device"].unique():
            device_df = df.loc[df["device"] == device]
            try:
                opposite_acc, opposite_count = opposite(
                    device_df.loc[device_df["condition"] == "basic"]["Accuracy"].item(),
                    device_df.loc[device_df["condition"] == "basic"]["count"].item(),
                    device_df.loc[device_df["condition"] == c]["Accuracy"].item(),
                    device_df.loc[device_df["condition"] == c]["count"].item(),
                )
            except ValueError:
                opposite_acc, opposite_count = None, 0

            row = pd.DataFrame(
                [
                    {
                        "device": device,
                        "condition": f"not_{c}",
                        "Accuracy": opposite_acc,
                        "count": opposite_count,
                    }
                ]
            )
            row = row.dropna(axis=1, how="all")
            df = pd.concat([df, row], ignore_index=True)

    category_dict = {
        "b_20": {"category": "Battery", "description": "Low\nBattery"},
        "not_b_20": {"category": "Battery", "description": "Normal\nBattery"},
        "b_15": {"category": "Battery", "description": "Battery level ≤ 15%"},
        "b_10": {"category": "Battery", "description": "Battery level ≤ 10%"},
        "c": {"category": "Charging", "description": "Charging"},
        "n_c": {"category": "Charging", "description": "Not\nCharging"},
        "b20_c": {
            "category": "Battery",
            "description": "Battery ≤ 20% & Charging",
        },
        "b20_n_c": {
            "category": "Battery",
            "description": "Battery ≤ 20%",
        },
        "ps": {"category": "Battery", "description": "Powersafe mode"},
        "idle": {"category": "Activity", "description": "LightIdle"},
        "d_i": {"category": "Activity", "description": "Inter-\nactive"},
        "d_u_i": {"category": "Activity", "description": "Uninter-\nactive"},
        "ram_115": {
            "category": "RAM Usage",
            "description": "High\nUsage",
        },
        "not_ram_115": {
            "category": "RAM Usage",
            "description": "Normal\nUsage",
        },
        "basic": {"category": "Baseline", "description": "Baseline"},
    }

    # only select some conditions
    number_of_traces = df.loc[df["condition"] == "basic"]["count"].sum()

    df["Weighted Accuracy"] = df["Accuracy"] * df["count"] / number_of_traces
    df = df.loc[
        df["condition"].isin(
            ["b_20", "not_b_20", "c", "n_c", "d_i", "d_u_i", "ram_115", "not_ram_115"]
        )
    ]

    df["Category"] = df["condition"].map(lambda x: category_dict[x]["category"])
    df["description"] = df["condition"].map(lambda x: category_dict[x]["description"])
    size = 16
    plt.rcParams.update(
        {
            "font.size": size,  # General font size for all elements
            "axes.labelsize": size,  # Axis label font size
            "axes.titlesize": 16,  # Title font size
            "xtick.labelsize": size,  # X-axis tick label font size
            "ytick.labelsize": size,  # Y-axis tick label font size
            "legend.fontsize": 15,  # Legend font size
        }
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        x="description",
        y="Accuracy",
        hue="Category",
        data=df,
        ax=ax,
        order=[
            "High\nUsage",
            "Normal\nUsage",
            "Inter-\nactive",
            "Uninter-\nactive",
            "Charging",
            "Not\nCharging",
            "Low\nBattery",
            "Normal\nBattery",
        ],
        showfliers=False,
    )
    ax.legend(loc="upper center", ncol=4)
    plt.xlabel("")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"figure5.pdf", format="pdf")
    plt.clf()
    return 0

if __name__ == '__main__':
    condition(True)
