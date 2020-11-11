#!/usr/bin/env bash
#
# Convenience script for requesting online prediction on AI Platform.
#
# Prerequisites:
#   - Google Cloud SDK
#
# Globals:
#   PROJECT_ID: Google Cloud project to use.
#
# Arguments:
#   $1: Path to file contained data for prediction in the format of:
#   a list of simple lists, each representing a data instance.
#   $2: Name of the model
#   $3: Version of the model

INPUT_DATA_FILE=$1
MODEL_NAME=$2
VERSION_NAME=$3

gcloud ml-engine predict \
--model $MODEL_NAME  \
--version $VERSION_NAME \
--json-instances $INPUT_DATA_FILE