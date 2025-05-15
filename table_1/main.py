"""Implementation of the majority voting evaluation."""

import pandas as pd
import numpy as np
from collections import Counter

def generate_results_dict(ks, dataset):
    result_dict = {}
    for k in ks:
        n_reps = max(500 // (k * 2), 5)
        accuracies = []
        accuracies_top_5 = []
        accuracies_top_10 = []
        for run in range(0, 10):
            acc_min = None
            acc_max = None
            df = pd.read_csv(f"results/{dataset}/results_{k}_traces_run_{run}.csv")
            corrects = 0
            corrects_top_5 = 0
            corrects_top_10 = 0
            for device in df["device_label"].unique():
                # take first k traces
                device_corrects = 0
                for i in range(n_reps):
                    np.random.seed(i)
                    traces = df[df["device_label"] == device].sample(k)
                    # get the majority of the top-1 predictions

                    if traces["top_1_prediction"].value_counts().idxmax() == device:
                        corrects += 1
                        device_corrects += 1

                    # get the majority of the top-n+1 predictions
                    predictions = traces[
                        [
                            "top_1_prediction",
                            "top_2_prediction",
                            "top_3_prediction",
                            "top_4_prediction",
                            "top_5_prediction",
                        ]
                    ].values
                    predictions = Counter(predictions.flatten())
                    most_common_devices = [
                        device for device, count in predictions.most_common(5)
                    ]
                    if device in most_common_devices:
                        corrects_top_5 += 1

                    predictions = traces[
                        [
                            "top_1_prediction",
                            "top_2_prediction",
                            "top_3_prediction",
                            "top_4_prediction",
                            "top_5_prediction",
                            "top_6_prediction",
                            "top_7_prediction",
                            "top_8_prediction",
                            "top_9_prediction",
                            "top_10_prediction",
                        ]
                    ].values
                    predictions = Counter(predictions.flatten())
                    most_common_devices = [
                        device for device, count in predictions.most_common(10)
                    ]
                    if device in most_common_devices:
                        corrects_top_10 += 1

                device_accuracy = device_corrects / n_reps
                # print('Device: ', device, 'Accuracy: ', device_accuracy)
                if acc_min is None or device_accuracy < acc_min:
                    acc_min = device_accuracy
                if acc_max is None or device_accuracy > acc_max:
                    acc_max = device_accuracy

            accuracy = corrects / (len(df["device_label"].unique()) * n_reps)
            accuracy_top_5 = corrects_top_5 / (len(df["device_label"].unique()) * n_reps)
            accuracy_top_10 = corrects_top_10 / (len(df["device_label"].unique()) * n_reps)
            accuracies.append(accuracy)
            accuracies_top_5.append(accuracy_top_5)
            accuracies_top_10.append(accuracy_top_10)

        result_dict[f"{k}_traces"] = {
            "top_1_accuracy": f"{100*np.mean(accuracies):.2f}%",
            "top_5_accuracy": f"{100*np.mean(accuracies_top_5):.2f}%",
            "top_10_accuracy": f"{100*np.mean(accuracies_top_10):.2f}%",
        }
    return result_dict


def generate_rows(result_dict, dataset):
    table = ""
    for k, v in result_dict.items():
        table += f"| {k} | {dataset} | {v['top_1_accuracy']} | {v['top_5_accuracy']} | {v['top_10_accuracy']} |\n"
    return table

if __name__ == "__main__":
    n_traces = [[8, 16, 32], [8, 16, 32], [64, 128, 256]]
    datasets = ["atomicinc_100", "drawnapart_100", "atomicinc_1000"]
    results_dicts = []
    with open("results_table.md", "w") as f:
        f.write("# Results Table\n")
        f.write("| Traces | Fingerprint | Top-1 Accuracy | Top-5 Accuracy | Top-10 Accuracy |\n")
        f.write("|--------|------------|----------------|----------------|----------------|\n")
        for ks, dataset in zip(n_traces, datasets):
            result_dict = generate_results_dict(ks, dataset)
            f.write(generate_rows(result_dict, dataset))
        f.write("\n")