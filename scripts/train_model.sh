#!/usr/bin/env bash
#
# Convenience script for running ML training jobs.
#
# Prerequisites:
#   - Google Cloud SDK
#
# Globals:
#   PROJECT_ID: Google Cloud project to use.
#   BUCKET_ID: Google Cloud Storage bucket to store output.
#
# Arguments:
#   $1: Path or BigQuery table to dataset for ML training and eval,
#       specified as PROJECT_ID.DATASET.TABLE_NAME.
#   $2: (Optional) Whether to run `local` (on-prem) or `remote` (GCP).
#   $3: (Optional) Whether to run `train` or `hptuning`.
#   $4: (Optional) additional arguments to pass to the trainer.

RUN_ENV=$1
EXTRA_TRAINER_ARGS="${@:2}"
IMAGE_URI=eu.gcr.io/$PROJECT_ID/trainer:0.1

if [[ ! "$RUN_ENV" =~ ^(local|remote)$ ]]; then
  RUN_ENV=local
fi

NOW="$(date +"%d%m_%H%M")"
JOB_PREFIX="train_model"
COMMIT_HASH="$(git rev-parse --verify HEAD)"
COMMIT_MESSAGE="$(git log -1 --pretty=%B)"
COMMIT_MESSAGE_WITHOUT_BLANK="${COMMIT_MESSAGE//[[:blank:]]/_}"
COMMIT_MESSAGE_WITHOUT_NEWLINE="${COMMIT_MESSAGE_WITHOUT_BLANK//$'\n'/_}"
JOB_NAME="${JOB_PREFIX}_${RUN_TYPE}_${NOW}_${COMMIT_MESSAGE_WITHOUT_NEWLINE////_}_${COMMIT_HASH}"
JOB_DIR="gs://$BUCKET_ID/models/$JOB_NAME"
PACKAGE_PATH=classifier
MAIN_TRAINER_MODULE=$PACKAGE_PATH.task
REGION=europe-west1
CONFIG_FILE=config/config.yaml #TODO different configs

# Specify arguments for remote (AI Platform) or local (on-premise) execution
echo "$RUN_ENV"
if [ "$RUN_ENV" = 'remote' ]; then

  if [[ -z $(git status -s --untracked-files=no) ]]; then
    echo "tree is clean"
  else
    echo "tree is dirty, please commit changes before training remotely"
    exit
  fi

  RUN_ENV_ARGS="jobs submit training $JOB_NAME \
    --region $REGION \
    --master-image-uri $IMAGE_URI \
    --config $CONFIG_FILE \
    "
else # assume `local`
  RUN_ENV_ARGS="local train \
	  --package-path $PACKAGE_PATH \
	  --module-name $MAIN_TRAINER_MODULE"
fi

# Specify arguments to pass to the trainer module (trainer/task.py)
TRAINER_ARGS="\
  --environment $RUN_ENV \
  --experiment_name $COMMIT_MESSAGE_WITHOUT_NEWLINE \
  --comet_api_key $COMET_API_KEY \
  "

CMD="gcloud ai-platform $RUN_ENV_ARGS \
  --job-dir $JOB_DIR \
  -- \
  $TRAINER_ARGS \
  $EXTRA_TRAINER_ARGS \
  "
if [ "$RUN_ENV" = 'remote' ]; then
  eval "docker build -f classifier.dockerfile -t $IMAGE_URI ./ && docker push $IMAGE_URI && $CMD"
else
  echo "Running command: $CMD"
  eval "$CMD"
fi
echo $JOB_DIR
