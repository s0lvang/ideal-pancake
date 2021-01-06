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

INPUT=$1
RUN_ENV=$2
RUN_TYPE=$3
EXTRA_TRAINER_ARGS=$4
IMAGE_URI=eu.gcr.io/$PROJECT_ID/trainer:0.1

if [[ ! "$RUN_ENV" =~ ^(local|remote)$ ]]; then
  RUN_ENV=local;
fi

if [[ ! "$RUN_TYPE" =~ ^(train|hptuning)$ ]]; then
  RUN_TYPE=train;
fi

NOW="$(date +"%d%m_%H%M")"
JOB_PREFIX="hardcore_ml_shit"
COMMIT_HASH="$(git rev-parse --verify HEAD)"
COMMIT_MESSAGE="$(git log -1 --pretty=%B)"
JOB_NAME="${RUN_TYPE}_${NOW}_${COMMIT_MESSAGE// /_}_${COMMIT_HASH}"
JOB_DIR="gs://$BUCKET_ID/models/$JOB_NAME"
PACKAGE_PATH=trainer
MAIN_TRAINER_MODULE=$PACKAGE_PATH.task
REGION=europe-west1

if [ "$RUN_TYPE" = 'hptuning' ]; then
  CONFIG_FILE=config/hptuning_config.yaml
else  # Assume `train`
  CONFIG_FILE=config/config.yaml
fi

# Specify arguments for remote (AI Platform) or local (on-premise) execution
echo "$RUN_ENV"
if [ "$RUN_ENV" = 'remote' ]; then
  RUN_ENV_ARGS="jobs submit training $JOB_NAME \
    --region $REGION \
    --master-image-uri $IMAGE_URI \
    --config $CONFIG_FILE \
    "
else  # assume `local`
  RUN_ENV_ARGS="local train \
	  --package-path $PACKAGE_PATH \
	  --module-name $MAIN_TRAINER_MODULE"
fi

# Specify arguments to pass to the trainer module (trainer/task.py)
TRAINER_ARGS="\
  --input $INPUT \
  "

CMD="gcloud ai-platform $RUN_ENV_ARGS \
  --job-dir $JOB_DIR \
  -- \
  $TRAINER_ARGS \
  $EXTRA_TRAINER_ARGS \
  "
kill $(lsof -ti tcp:6006) # Kill tensoboard if it runs
tensorboard --logdir="$JOB_DIR/tensorboard" &
open "http://localhost:6006"
if [ "$RUN_ENV" = 'remote' ]; then
	eval "docker build -f Dockerfile -t $IMAGE_URI ./ && docker push $IMAGE_URI && $CMD"
else
echo "Running command: $CMD"
eval "$CMD"
fi
echo $JOB_DIR

