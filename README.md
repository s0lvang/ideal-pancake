# Ideal Pancake

## Setup

### Prerequisites

`python3.7`

`google-cloud-sdk`

`poetry`

### Gettings started

Clone the project.

In credentials/ you need two files:
`ideal-pancake-docker-id.json`
`ideal-pancake-google-application-credentials.json`

These are created by creating service accounts on `cloud.google.com`. Get in touch with us if you don't have the access you should have. You will need to rename the files that are generated appropriately.

### Installing

`poetry install`

`. ./credentials/env.sh`

### Extracting features

`poetry run ./scripts/generate_features.sh [environment] [dataset] --download_files [bool]`
Environment has the possible values `remote` or `local`. Which decides if a job should be run locally or dispatched as a job to gcp
dataset has the possible values `["emip", "jetris", "emip-images", "mooc-images", "cscw", "fractions"]`
download_files decides if the dataset should be downloaded or local files should be used

If the feature extraction is run remotely, the features will be saved to a bucket which the classifier can fetch

### Training the model

`poetry run ./scripts/train_model.sh [environment] [in_study_dataset] [out_of_study_dataset]`

in_study_dataset and out_of_study has the possible values `["emip", "jetris", "emip-images", "mooc-images", "cscw", "fractions"]`
This script fetches the premade features from gcs and trains on them

#### Examples

` poetry run ./scripts/generate_features.sh remote emip --download_files True`
` poetry run ./scripts/train_model.sh local emip cscw`
