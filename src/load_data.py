import os

from datasets import load_dataset, load_from_disk

def load_dataset_from_huggingface():
    ds = load_dataset("RomanTeucher/text2cypher-curated")
    ds.save_to_disk("./../dataset")
    return ds

def load_data():
    path = "./../dataset"
    if os.path.exists(path):
        ds = load_from_disk(path)
    else: ds = load_dataset_from_huggingface()
    return ds