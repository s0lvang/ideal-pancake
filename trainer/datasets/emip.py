import pandas as pd
from trainer import globals
from itertools import takewhile


def prepare_emip_files(file_references, metadata_references):
    labels = pd.Series()
    dataset = pd.DataFrame()
    with metadata_references[0].open("r") as f:
        metadata_file = pd.read_csv(f)
    for file_reference in file_references:
        with file_reference.open("r") as f:
            dataset, labels = prepare_emip_file(f, metadata_file, dataset, labels)
    # dataset = dataset.rename(columns={"gameID": "id", "time[milliseconds]": "Time"})
    return dataset, labels


def prepare_emip_file(f, metadata_file, dataset, labels):
    if globals.dataset.name == "emip":
        c = globals.dataset.config
    else:
        print(globals)
        c = globals.out_of_study_dataset

    subject_id = get_header(f)["Subject"][0]
    csv = pd.read_csv(f, sep="\t", comment="#")
    csv["id"] = int(subject_id)
    dataset = dataset.append(csv, ignore_index=True)
    labels.at[int(subject_id)] = metadata_file.loc[int(subject_id) - 1, c.LABEL]
    return dataset, labels


def get_header(file):
    headiter = takewhile(lambda s: s.startswith("##"), file)
    headerList = list(map(lambda x: x.strip("##").strip().split(":"), headiter))
    header = dict(filter(lambda x: len(x) == 2, headerList))
    split_on_tab = lambda x: x.split("\t")[1:]
    header = {k: split_on_tab(v) for k, v in header.items()}
    file.seek(0, 0)
    return header