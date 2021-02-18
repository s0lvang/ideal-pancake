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

### Starting

`poetry run ./scripts/train.sh [environment] [in_study_dataset] [out_of_study_dataset] --generate_features [bool] --download_files [bool]`

Environment has the possible values `remote` or `local`. Which decides if a job should be run locally or dispatched as a job to gcp

in_study_dataset and out_of_study has the possible values `["emip", "jetris", "emip-images", "mooc-images]`

Generate features decides if the features should be generated or premade features should be downloaded from gcp

download_files decides if the dataset should be downloaded or local files should be used

#### Example

` poetry run ./scripts/train.sh local emip emip --generate_features true --download_files false`
