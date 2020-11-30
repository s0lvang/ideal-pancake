import pandas as pd
from trainer.datasets import datasets
from trainer import config


def prepare_emip_files(file_references, metadata_references):
    labels = pd.Series()
    dataset = pd.DataFrame()
    with datasets.get_files(metadata_references[0]) as f:
        metadata_file = pd.read_csv(f)
    for file_reference in file_references:
        with datasets.get_files(file_reference) as f:
            dataset, labels = prepare_emip_file(f, metadata_file, dataset, labels)
    # dataset = dataset.rename(columns={"gameID": "id", "time[milliseconds]": "Time"})
    return dataset, labels


def prepare_emip_file(f, metadata_file, dataset, labels):
    subject_id = datasets.get_header(f)["Subject"][0]
    csv = pd.read_csv(f, sep="\t", comment="#")
    csv["id"] = int(subject_id)
    dataset = dataset.append(csv, ignore_index=True)
    labels.at[int(subject_id)] = metadata_file.loc[int(subject_id) - 1, config.LABEL]
    return dataset, labels
