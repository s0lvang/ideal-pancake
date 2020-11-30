import pandas as pd
from trainer.datasets import datasets


def prepare_jetris_files(file_references):
    labels = pd.Series()
    dataset = pd.DataFrame()
    for file_reference in file_references:
        with datasets.get_files(file_reference) as f:
            dataset, labels = prepare_jetris_file(f, dataset, labels)
    # labels = convert_labels_to_categorical()
    dataset = dataset.rename(columns={"gameID": "id", "time[milliseconds]": "Time"})
    return dataset, labels


def prepare_jetris_file(f, dataset, labels):
    csv = pd.read_csv(f, comment="#")
    csv = csv[
        csv["Pupil.initial"] != "saccade"
    ]  # this drops all lines that are saccades, we should do something smarter here.
    game_id = csv["gameID"][0]
    dataset = dataset.append(csv, ignore_index=True)
    labels.at[int(game_id)] = csv["Score.1"].iloc[-1]
    return dataset, labels
