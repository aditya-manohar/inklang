import pandas as pd
import os
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def inkl_print(msg: str) -> None:
    logging.debug(f"inkl_print: {msg}")
    print(f"[inklang] {msg}")

def load_dataset(path: str):
    try:
        full_path = os.path.join("uploads", path)
        logging.debug(f"Loading dataset from '{full_path}'")
        if path.endswith('.csv'):
            df = pd.read_csv(full_path)
        elif path.endswith('.json'):
            df = pd.read_json(full_path)
        else:
            raise ValueError(f"[inklang] Unsupported file format: {path}")
        if 'label' not in df.columns:
            raise ValueError(f"[inklang] Dataset must have a 'label' column. Found columns: {list(df.columns)}")
        logging.debug(f"Loaded dataset from '{full_path}' rows={len(df)} cols={len(df.columns)}")
        inkl_print(f"Loaded dataset from '{full_path}' rows={len(df)} cols={len(df.columns)}")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        inkl_print(f"Error loading dataset: {str(e)}")
        return None