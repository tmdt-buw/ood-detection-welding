from pathlib import Path
import numpy as np
import pandas as pd


def export_to_npy_files(data_path: str | Path):
    df = pd.read_csv(data_path / "welding_data.csv")

    labels, exp_ids, welding_run_ids = df["label"].values, df["exp_id"].values, df["welding_run_id"].values
    data = df.drop(columns=["label", "exp_id", "welding_run_id"])
    
    cols_v = data.columns[data.columns.str.startswith("V")]
    cols_i = data.columns[data.columns.str.startswith("I")]
    
    current_data = data[cols_i].values
    voltage_data = data[cols_v].values

    data = np.stack([current_data, voltage_data], axis=2)
    ids = np.stack([np.arange(len(exp_ids)), exp_ids, welding_run_ids, labels], axis=1)

    np.save(data_path / "welding_data.npy", data)
    np.save(data_path / "welding_ids.npy", ids)
    print(f"Saved to {data_path}")



if __name__ == "__main__":
    export_to_npy_files()