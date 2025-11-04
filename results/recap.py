import os
import pandas as pd
import glob
import ast  # Per convertire le stringhe in dizionari/liste
import numpy as np
import scipy.stats as st

base_path = "./"
folder_exps = "25_MIT_0128_B"
name_exps = "chengdu_0128_linear___16_*"
folder_paths = glob.glob(os.path.join(base_path, folder_exps, name_exps))

folder_model = "AE"
folder_analysis = "copulaLatent_data_analysis"
filename = "metrics_compare__copulaLatent_data.csv"
data_list = []
do_copula = False
do_ea = True
if do_ea:
    output_file = os.path.join(base_path,folder_exps, f"all_metrics_combined__{folder_model}_{folder_analysis}.csv")
elif do_copula:
    output_file = os.path.join(base_path,folder_exps, f"all_metrics_combined__{folder_model}_{folder_analysis}_cop.csv")

for folder_path in folder_paths:
    print(folder_path)
    folder_name = os.path.basename(folder_path)  # Nome della cartella
    csv_file = os.path.join(folder_path, folder_model, folder_analysis, filename)

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)

        # Crea un dizionario con la struttura {metric: (real_cop, real_ae)}
        metrics_data = {"folder": folder_name}
        for _, row in df.iterrows():
            metric = row["metric"]

            if do_ea:
                real_ae = row["real_ae"]
                try:
                    real_ae = ast.literal_eval(real_ae)
                except (ValueError, SyntaxError):
                    pass
                if metric == "wasserstein":
                    metrics_data[f"{metric}_mean"] = real_ae[0]
                    metrics_data[f"{metric}_sum"] = sum(real_ae[1])
                    metrics_data[f"{metric}_list"] = real_ae[1]

                elif isinstance(real_ae, dict):
                    for key, value in real_ae.items():
                        metrics_data[f"{metric}_{key}"] = value
                else:
                    metrics_data[f"{metric}"] = real_ae




            if do_copula:
                real_cop = row["real_cop"]
                try:
                    real_cop = ast.literal_eval(real_cop)
                except (ValueError, SyntaxError):
                    pass
                if metric == "wasserstein":
                    metrics_data[f"{metric}_mean"] = real_cop[0]
                    metrics_data[f"{metric}_sum"] = sum(real_cop[1])
                    metrics_data[f"{metric}_list"] = real_cop[1]

                elif isinstance(real_cop, dict):
                    for key, value in real_cop.items():
                        metrics_data[f"{metric}_{key}"] = value
                else:
                    metrics_data[f"{metric}"] = real_cop


        data_list.append(metrics_data)


if data_list:
    df = pd.DataFrame(data_list)
    df.to_csv(output_file, index=False)
    print(f"File salvato: {output_file}")
else:
    print(f"Nessun file valido trovato. {output_file}")
