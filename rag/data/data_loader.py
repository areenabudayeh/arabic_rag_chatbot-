from datasets import load_dataset
import pandas as pd

def load_arcd_dataset():
    """Load ARCD dataset from Hugging Face"""
    dataset = load_dataset("hsseinmz/arcd")
    
    df_train = pd.DataFrame(dataset["train"])
    df_val = pd.DataFrame(dataset["validation"])
    
    return df_train, df_val

def get_dataset_info(df_train, df_val):
    """Get basic information about the datasets"""
    info = {
        "train_size": len(df_train),
        "val_size": len(df_val),
        "train_missing": df_train.isnull().sum().to_dict(),
        "val_missing": df_val.isnull().sum().to_dict()
    }
    return info
