from pathlib import Path
import numpy as np
import pandas as pd


def export_to_npy_files(data_path: str | Path):
    """
    Export welding data from CSV to NumPy binary files for efficient loading.
    
    This function reads welding data from a CSV file and processes it into two NumPy arrays:
    1. A 3D data array containing current and voltage measurements
    2. An ID array containing indices, experiment IDs, welding run IDs, and labels
    
    The function expects the CSV to have columns starting with 'V' (voltage) and 'I' (current),
    plus 'label', 'exp_id', and 'welding_run_id' columns.
    
    Parameters
    ----------
    data_path : str or Path
        Path to the directory containing 'welding_data.csv' file.
        The output .npy files will also be saved to this directory.
    
    Saves
    -----
    welding_data.npy : ndarray of shape (n_samples, n_timepoints, 2)
        3D array where the last dimension contains [current_data, voltage_data]
    welding_ids.npy : ndarray of shape (n_samples, 4)
        2D array with columns [sample_index, exp_id, welding_run_id, label]
    
    Raises
    ------
    FileNotFoundError
        If 'welding_data.csv' is not found in the specified data_path
    KeyError
        If required columns ('label', 'exp_id', 'welding_run_id') are missing from CSV
    
    Examples
    --------
    >>> export_to_npy_files("path/to/welding/data")
    Saved to path/to/welding/data
    
    >>> from pathlib import Path
    >>> export_to_npy_files(Path("data/Welding"))
    Saved to data/Welding
    """
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
    export_to_npy_files(Path(__file__).parent)