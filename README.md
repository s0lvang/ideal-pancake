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

### Installing & Starting

`poetry install`

`./credentials/env.sh`

`poetry run scripts/train.sh temp local` or `poetry run scripts/train temp remote`
