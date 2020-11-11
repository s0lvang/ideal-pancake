#!/usr/bin/env bash
# Convenience script for deploying trained scikit-learn model.
#
# Prerequisites:
#   - Google Cloud SDK
#
# Globals:
#   PROJECT_ID: Google Cloud project to use.
#
# Arguments:
#   $1: Path to directory containing trained and exported scikit-learn model
#   $2: Name of the model to be deployed
#   $3: Version of the model to be deployed

MODEL_DIR=$1
MODEL_NAME=$2
VERSION_NAME=$3

REGION=us-central1
FRAMEWORK=SCIKIT_LEARN
RUN_TIME=1.13
PYTHON_VERSION=3.5 # only support python 2.7 and 3.5

if gcloud ml-engine models list | grep "$MODEL_NAME" &> /dev/null
then
   echo "Model already exists."
else
    # 1. Create model
    gcloud ml-engine models create "$MODEL_NAME" \
    --regions=$REGION
fi


if gcloud ml-engine versions list --model="$MODEL_NAME" | grep "$VERSION_NAME" &> /dev/null
then
   echo "Version already exists."
else
    # 2. Create version
    gcloud ml-engine versions create "$VERSION_NAME" \
    --model "$MODEL_NAME" \
    --origin "$MODEL_DIR" \
    --framework "$FRAMEWORK" \
    --runtime-version="$RUN_TIME" \
    --python-version="$PYTHON_VERSION"
fi
