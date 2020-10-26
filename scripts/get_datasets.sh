#!/bin/bash

mkdir datasets
cd datasets
directory_name = "eeg-during-mental-arithmetic-tasks"
file_name = "eeg-during-mental-arithmetic-tasks.zip"
curl "https://physionet.org/static/published-projects/eegmat/eeg-during-mental-arithmetic-tasks-1.0.0.zip" -o $file_name
unzip $file_name -d $directory_name
